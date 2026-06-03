import json
from pathlib import Path

input = "fr_oral_sl"
output=f"{input}.json"

lessons_folder = Path("pdf") / input
output_path = Path("pdf")/output

def flatten(data):
    lesson = data.get("lesson") or {}
    return {
        "lesson_number": lesson.get("lesson_number"),
        "title": lesson.get("title", ""),
        "topic": lesson.get("topic", ""),
        "objective": lesson.get("objective", ""),
        "page_start": data.get("page_start"),
        "page_end": data.get("page_end"),
        "sections": data.get("sections", []),
    }

def sort_key(lesson):
    num = lesson.get("lesson_number")
    try:
        return (0, int(num))
    except (TypeError, ValueError):
        return (1, str(num))

def main():
    lesson_files = sorted(lessons_folder.glob("lesson_*.json"))
    if not lesson_files:
        print(f"No lesson_*.json files found in {lessons_folder}")
        return

    lessons = sorted(
        (flatten(json.loads(p.read_text(encoding="utf-8"))) for p in lesson_files),
        key=sort_key,
    )

    output_path.write_text(
        json.dumps({"lessons": lessons}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Merged {len(lessons)} lessons -> {output_path}")
    print("Next: python indexer.py")

if __name__ == "__main__":
    main()
