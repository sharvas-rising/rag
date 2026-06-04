import json
from pathlib import Path
from openai import OpenAI
import requests
from dotenv import load_dotenv
import os


# Configuration
INPUT_FILE = Path(__file__).parent / "pdf" / "fr_letter_sl" / "fr_letter_sl.json"
SUBJECT = "FR"
LEVEL = "Letter"

load_dotenv()

env_file = Path(__file__).parent / ".env"
api_key = None
if env_file.exists():
    with open(env_file, "r") as f:
        for line in f:
            if line.startswith("OPENAI_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
                break

client = OpenAI(api_key=api_key)

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
if not supabase_url or not supabase_key:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")

supabase_url = supabase_url.rstrip("/")


def load_lessons():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def get_embedding(text):
    resp = client.embeddings.create(input=text, model="text-embedding-3-small")
    return resp.data[0].embedding

def index_lessons():
    data = load_lessons()
    cursor_count = 0

    for lesson in data["lessons"]:
        for section in lesson.get("sections", []):
            rich_text = f"{lesson['topic']}. {lesson['objective']}. {section['section_name']}: {section['content']}"
            embedding = get_embedding(rich_text)
            chunk_id = f"lesson_{lesson['lesson_number']}_level_{LEVEL.lower()}_section_{section['section_name'].replace(' ', '_').lower()}"

            payload = {
                "id": chunk_id,
                "lesson_number": str(lesson["lesson_number"]),
                "subject": SUBJECT,
                "level": LEVEL,
                "title": lesson.get("title", ""),
                "topic": lesson["topic"],
                "objective": lesson["objective"],
                "section_name": section["section_name"],
                "duration": section.get("duration", ""),
                "content": section["content"],
                "embedding": embedding,
            }

            try:
                resp = requests.post(
                    f"{supabase_url}/rest/v1/lessons_chunks",
                    json=payload,
                    headers={
                        "apikey": supabase_key,
                        "Authorization": f"Bearer {supabase_key}",
                        "Content-Type": "application/json",
                        "Prefer": "resolution=merge-duplicates",
                    },
                )
                if resp.status_code not in (200, 201):
                    print(f"  ❌ [{chunk_id}] {resp.status_code}: {resp.text}")
                else:
                    cursor_count += 1
                    print(f"  [{cursor_count}] {chunk_id}")
            except Exception as e:
                print(f"  ❌ [{chunk_id}] {e}")

    print(f"\n✅ Indexed {cursor_count} chunks")

if __name__ == "__main__":
    index_lessons()
