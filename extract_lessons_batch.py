import csv
import json
import base64
import os
import time
from collections import defaultdict
from pathlib import Path
from io import BytesIO
import pdfplumber
from openai import OpenAI
from dotenv import load_dotenv

os.system("cls")

# Read .env file directly
env_file = Path(__file__).parent / ".env"
api_key = None
if env_file.exists():
    with open(env_file, "r") as f:
        for line in f:
            if line.startswith("OPENAI_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
                break

if api_key:
    print(f"API Key loaded: {api_key[:20]}...{api_key[-10:]}")
else:
    print("ERROR: OPENAI_API_KEY not found in .env file")

pdf_path = Path("pdf") / "fr_oral_sl.pdf"
csv_path = Path("pdf") / "fr_oral_sl.csv"
output_folder = Path("pdf") / "fr_oral_sl"
output_folder.mkdir(exist_ok=True)

client = None  # initialized lazily in main() so CSV/PDF errors surface before SSL init

extraction_prompt = """You are extracting structured curriculum content from a scanned lesson-page image.

Your task is to analyze the page carefully and return CLEAN JSON.

IMPORTANT RULES:
- Preserve the educational hierarchy exactly.
- Identify:
  - lesson information
  - section blocks
  - page number
  - visible text content
- Ignore decorative graphics and illustrations unless they contain text.
- Keep text exactly as written where possible.
- Merge multiline text naturally WHILE PRESERVING READING ORDER.
- CRITICAL: Read and preserve text in strict top-to-bottom, left-to-right order.
- Do NOT scramble or reorder text within sections.
- Do not hallucinate missing text.
- Output ONLY valid JSON.
- No markdown.
- No explanations.

JSON SCHEMA:

{
  "lesson": {
    "lesson_number": "",
    "title": "",
    "topic": "",
    "objective": ""
  },
  "page_number": "",
  "sections": [
    {
      "section_name": "",
      "duration": "",
      "content": ""
    }
  ]
}

EXTRACTION RULES:

1. LESSON
Extract:
- lesson number
- lesson title
- topic
- objective

Example:
"Topic: Simple Conversations"
"Objective: Participate in simple conversations (greetings)."

2. PAGE NUMBER
Extract the printed page number visible on the page.

3. SECTION DETECTION
Treat each pedagogical block as a section.

ALLOWED section names — use ONE of these EXACTLY, even if the document phrases it slightly differently (e.g. "Checking for Understanding - Point & Say" -> "Check for Understanding"):
- Warm Up
- Starter
- I do - Key Learning Points
- Check for Understanding
- We do - Model and Practice
- You do - Independent Practice
- Turn and Talk

Do NOT invent other section names. If a block clearly doesn't match any of the above, omit it.

4. CONTENT
For each section READ STRICTLY TOP-TO-BOTTOM, LEFT-TO-RIGHT:
- Read text in exact visual order (top line first, then next line below)
- If text flows across columns, read entire left column top-to-bottom, then entire right column
- Include all dialogue, prompts, questions, vocabulary, answers in the order they appear
- Preserve exact sentence sequence — do NOT reorder for readability
- Combine sentences naturally while keeping their original order
- Teaching content flow depends on correct ordering

5. DURATION
Extract visible duration if present:
Examples:
- "5 min"
- "10-15 min"
- "1 min"

6. OUTPUT QUALITY
- content should be a single clean string
- CRITICAL: Preserve strict text ordering (top-to-bottom, left-to-right)
- Do NOT scramble or reorder sentences within sections
- Do NOT rewrite for better grammar — keep exact order even if awkward
- Preserve conversational flow by keeping question-answer pairs in sequence

EXAMPLE OUTPUT:

{
  "lesson": {
    "lesson_number": "1",
    "title": "Lesson 1",
    "topic": "Simple Conversations",
    "objective": "Participate in simple conversations (greetings)."
  },
  "page_number": "12",
  "sections": [
    {
      "section_name": "Warm Up",
      "duration": "3-5 min",
      "content": "Hello Song! Good morning, good afternoon, good evening."
    },
    {
      "section_name": "I do — Key Learning Points",
      "duration": "10-15 min",
      "content": "Good morning. Good afternoon. Good evening. Hello. Goodbye."
    }
  ]
}
"""

def encode_image_to_base64(pil_image):
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.standard_b64encode(buffer.read()).decode("utf-8")

def create_batch_request(tasks):
    """Create a batch request for a group of (lesson_number, page_num, pil_image) tasks."""
    requests = []

    for lesson_number, page_num, pil_image in tasks:
        img_base64 = encode_image_to_base64(pil_image)
        print(f"  Prepared lesson {lesson_number} page {page_num}")

        requests.append({
            "custom_id": f"lesson-{lesson_number}-page-{page_num}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "max_tokens": 2000,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": extraction_prompt
                            }
                        ]
                    }
                ]
            }
        })

    return requests

def submit_batch(requests):
    """Submit batch to OpenAI and return batch ID."""
    # Write JSONL to temp file
    jsonl_path = Path("batch_input.jsonl")
    with open(jsonl_path, "w") as f:
        for r in requests:
            f.write(json.dumps(r) + "\n")

    # Upload file
    with open(jsonl_path, "rb") as f:
        batch_input_file = client.files.create(
            file=f,
            purpose="batch"
        )

    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    print(f"Batch submitted: {batch.id}")
    return batch.id

def wait_for_batch(batch_id):
    """Poll batch status until complete."""
    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"Batch {batch_id}: {batch.status}")

        if batch.status == "completed":
            return batch
        elif batch.status == "failed":
            print(f"Batch failed!")
            print(f"  Status: {batch.status}")
            print(f"  Full batch object: {batch}")
            return None

        time.sleep(5)  # Check every 5 seconds

def process_batch_results(batch_id):
    """Retrieve and parse batch results. Returns dict[custom_id] -> data."""
    batch = client.batches.retrieve(batch_id)

    if batch.status != "completed":
        print(f"Batch {batch_id} not completed")
        return None

    print(f"  request_counts: {batch.request_counts}")

    # Check for error file
    if batch.error_file_id:
        error_content = client.files.content(batch.error_file_id)
        print(f"  Error file contents:\n{error_content.text[:2000]}")

    if not batch.output_file_id:
        print(f"  No output file — all requests may have failed. Check error output above.")
        return None

    # Get output file
    raw = client.files.content(batch.output_file_id)

    results = {}
    for line in raw.text.strip().split('\n'):
        if not line:
            continue
        result = json.loads(line)
        custom_id = result["custom_id"]

        try:
            response_text = result["response"]["body"]["choices"][0]["message"]["content"]

            # Strip markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            data = json.loads(response_text)
            results[custom_id] = data
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing {custom_id}: {e}")
            results[custom_id] = None

    return results

def read_lessons_csv(path):
    """Read CSV/TSV with columns lesson_number, page_start, page_end.

    Returns list of (lesson_number, page_start, page_end) ints, in file order.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        delimiter = "\t" if "\t" in sample.splitlines()[0] else ","
        reader = csv.DictReader(f, delimiter=delimiter)
        lessons = []
        for row in reader:
            lesson_number = int(row["lesson_number"])
            page_start = int(row["page_start"])
            page_end = int(row["page_end"])
            lessons.append((lesson_number, page_start, page_end))
    return lessons

CANONICAL_SECTIONS = [
    "Warm Up",
    "Starter",
    "I do - Key Learning Points",
    "Check for Understanding",
    "We do - Model and Practice",
    "You do - Independent Practice",
    "Turn and Talk",
]

def normalize_section_name(name):
    """Map any model-returned section name to one of the canonical names.

    Returns the canonical name, or the original string if no rule matches
    (so the user can spot drift rather than silently losing the section).
    """
    if not name:
        return name
    n = name.lower()
    if "warm" in n:
        return "Warm Up"
    if "starter" in n:
        return "Starter"
    if "i do" in n or "key learning" in n:
        return "I do - Key Learning Points"
    if "check" in n and "understand" in n:
        return "Check for Understanding"
    if "we do" in n or "model and practice" in n:
        return "We do - Model and Practice"
    if "you do" in n or "independent" in n:
        return "You do - Independent Practice"
    if "turn" in n and "talk" in n:
        return "Turn and Talk"
    return name

def write_lesson_json(lesson_number, page_data_pairs):
    """Write one markdown file per lesson as a single merged JSON object.

    page_data_pairs: list of (page_num_from_csv, data) in page order.
    Sections from all pages are concatenated, each tagged with its page_number.
    lesson_number comes from the CSV; title/topic/objective use the first
    non-empty value seen across pages.
    """
    merged_lesson = {
        "lesson_number": str(lesson_number),
        "title": "",
        "topic": "",
        "objective": "",
    }
    merged_sections = []
    page_nums = []
    failed_pages = []

    for page_num, data in page_data_pairs:
        page_nums.append(page_num)
        if not data:
            failed_pages.append(page_num)
            continue

        page_lesson = data.get("lesson") or {}
        for field in ("title", "topic", "objective"):
            if not merged_lesson[field] and page_lesson.get(field):
                merged_lesson[field] = page_lesson[field]

        for section in data.get("sections", []) or []:
            normalized = dict(section)
            normalized["section_name"] = normalize_section_name(section.get("section_name", ""))
            merged_sections.append({"page_number": str(page_num), **normalized})

    merged = {
        "lesson": merged_lesson,
        "page_start": min(page_nums) if page_nums else None,
        "page_end": max(page_nums) if page_nums else None,
        "sections": merged_sections,
    }
    if failed_pages:
        merged["failed_pages"] = failed_pages

    out_path = output_folder / f"lesson_{lesson_number}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"Saved to {out_path}")

def main():
    global client

    print(f"Reading {csv_path}...")
    lessons = read_lessons_csv(csv_path)
    print(f"Loaded {len(lessons)} lessons from CSV")

    print(f"Initializing OpenAI client...")
    client = OpenAI(api_key=api_key)

    print(f"Opening {pdf_path}...")
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"Total pages in PDF: {total_pages}")

        # Build full task list: (lesson_number, page_num, pil_image)
        # Skip pages outside PDF range.
        tasks = []
        for lesson_number, page_start, page_end in lessons:
            for page_num in range(page_start, page_end + 1):
                if not (1 <= page_num <= total_pages):
                    print(f"  Warning: lesson {lesson_number} page {page_num} out of PDF range, skipping")
                    continue
                pil_image = pdf.pages[page_num - 1].to_image(resolution=150)
                tasks.append((lesson_number, page_num, pil_image))

        print(f"Total pages to process: {len(tasks)}")

        # Process in chunks, accumulating results across all batches.
        batch_size = 100
        batch_num = 1
        all_results = {}

        for chunk_start in range(0, len(tasks), batch_size):
            chunk = tasks[chunk_start:chunk_start + batch_size]
            preview = [(ln, pn) for ln, pn, _ in chunk]
            print(f"\n--- Batch {batch_num}: {preview} ---")

            requests = create_batch_request(chunk)
            batch_id = submit_batch(requests)

            batch = wait_for_batch(batch_id)
            if batch:
                results = process_batch_results(batch_id)
                if results:
                    all_results.update(results)

            batch_num += 1

    # Group results by lesson and write one markdown per lesson, in CSV order.
    by_lesson = defaultdict(list)
    for lesson_number, page_num, _ in tasks:
        custom_id = f"lesson-{lesson_number}-page-{page_num}"
        data = all_results.get(custom_id)
        by_lesson[lesson_number].append((page_num, data))

    for lesson_number, _, _ in lessons:
        if lesson_number in by_lesson:
            write_lesson_json(lesson_number, by_lesson[lesson_number])

    print(f"\nDone! Results saved to {output_folder}")

if __name__ == "__main__":
    main()
