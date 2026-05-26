"""Lesson Query API - FastAPI Server"""

import json
import logging
from functools import lru_cache
from difflib import get_close_matches
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from openai import OpenAI
from dotenv import load_dotenv
import os
# from langfuse import Langfuse
# from langfuse.openai import openai_integration

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
# LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
# LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")

if not OPENAI_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY not found in .env")

# langfuse = Langfuse(
#     public_key=LANGFUSE_PUBLIC_KEY,
#     secret_key=LANGFUSE_SECRET_KEY
# ) if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY else None
#
# if langfuse:
#     openai_integration()
#     logger.info("✓ Langfuse integration initialized")

langfuse = None

logger.info(f"✓ API Key loaded: {OPENAI_KEY[:20]}...{OPENAI_KEY[-10:]}")

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
MATCH_COUNT = 7
FUZZY_CUTOFF = 0.3

SYSTEM_PROMPT = (
    "You are a friendly teaching buddy helping teachers in Sub-Saharan Africa 🌍💚. "
    "Their English may be basic, so always:\n"
    "- Use very simple, clear English (short sentences, common words)\n"
    "- Be warm and encouraging, like a supportive peer — not a formal expert\n"
    "- Add relevant emojis to make it feel friendly and easy to read ✨\n"
    "- Keep answers short and to the point\n"
    "Answer only using the lesson content provided."
)

client = OpenAI(api_key=OPENAI_KEY)
session = requests.Session()
session.headers.update({
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
})

_catalog = None
_topic_to_lesson = None
_lesson_sections = None


# def log_trace(trace_name, metadata=None, input_data=None):
#     """Create a trace in Langfuse."""
#     if not langfuse:
#         return None
#     return langfuse.trace(name=trace_name, input=input_data, metadata=metadata or {})
#
#
# def log_span(trace, span_name, metadata=None, input_data=None):
#     """Create a span in a trace."""
#     if not trace:
#         return None
#     return trace.span(name=span_name, input=input_data, metadata=metadata or {})

def log_trace(trace_name, metadata=None, input_data=None):
    return None

def log_span(trace, span_name, metadata=None, input_data=None):
    return None


def _supabase_get(endpoint, params=None):
    """GET from Supabase REST API."""
    # span = None
    # if langfuse and langfuse.get_current_trace():
    #     span = langfuse.get_current_trace().span(name=f"supabase_get_{endpoint}", input={"params": params})

    resp = session.get(
        f"{SUPABASE_URL}/rest/v1/{endpoint}",
        params=params,
        headers={"Accept": "application/json"},
        timeout=10
    )
    resp.raise_for_status()
    result = resp.json()

    # if span:
    #     span.end(output={"row_count": len(result) if isinstance(result, list) else 1})

    return result


def _supabase_post(endpoint, json_data):
    """POST to Supabase RPC."""
    # span = None
    # if langfuse and langfuse.get_current_trace():
    #     span = langfuse.get_current_trace().span(name=f"supabase_post_{endpoint}", input={"data_keys": list(json_data.keys())})

    resp = session.post(
        f"{SUPABASE_URL}/rest/v1/{endpoint}",
        json=json_data,
        headers={"Content-Type": "application/json"},
        timeout=10
    )
    resp.raise_for_status()
    result = resp.json()

    # if span:
    #     span.end(output={"row_count": len(result) if isinstance(result, list) else 1})

    return result


def check_wa_id_24h(wa_id):
    """Check if wa_id has a record within past 24 hours. Returns subject/level or -1."""
    try:
        rows = _supabase_get("user_access_log", params={
            "wa_id": f"eq.{wa_id}",
            "select": "subject,level,access_time"
        })

        if rows:
            record = rows[0]
            access_time = datetime.fromisoformat(record["access_time"].replace("Z", "+00:00"))
            if datetime.now(access_time.tzinfo) - access_time < timedelta(hours=24):
                return {"subject": record["subject"], "level": record["level"]}

        return -1
    except Exception as e:
        logger.error(f"Error checking wa_id {wa_id}: {e}")
        return -1


def update_wa_id_access(wa_id, subject=None, level=None):
    """Update or insert user access record."""
    try:
        _supabase_post("user_access_log", {
            "wa_id": wa_id,
            "access_time": datetime.utcnow().isoformat(),
            "subject": subject,
            "level": level,
        })
        logger.info(f"Updated access log for wa_id: {wa_id}")
    except Exception as e:
        logger.error(f"Error updating access log for {wa_id}: {e}")


def _ensure_catalog():
    """Load catalog from Supabase (lazy-load)."""
    global _catalog, _topic_to_lesson, _lesson_sections

    if _catalog is not None:
        return _catalog, _topic_to_lesson, _lesson_sections

    logger.info("Loading catalog from Supabase...")
    rows = _supabase_get("lessons_chunks", params={
        "select": "lesson_number,topic,section_name,objective"
    })

    catalog = {field: set() for field in ["lesson_number", "topic", "section_name", "objective"]}
    topic_to_lesson = {}
    lesson_sections = {}

    for row in rows:
        for field in catalog:
            if row.get(field):
                catalog[field].add(row[field])

        if row.get("topic") and row.get("lesson_number"):
            topic_to_lesson[row["topic"]] = row["lesson_number"]

        if row.get("lesson_number") and row.get("section_name"):
            lesson_num = row["lesson_number"]
            if lesson_num not in lesson_sections:
                lesson_sections[lesson_num] = set()
            lesson_sections[lesson_num].add(row["section_name"])

    _catalog = {k: sorted(v) for k, v in catalog.items()}
    _topic_to_lesson = topic_to_lesson
    _lesson_sections = {k: sorted(v) for k, v in lesson_sections.items()}

    logger.info(f"Catalog loaded: {len(_catalog['lesson_number'])} lessons")
    return _catalog, _topic_to_lesson, _lesson_sections


@lru_cache(maxsize=128)
def _get_embedding(text):
    """Generate and cache embedding."""
    return client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    ).data[0].embedding


def _parse_json(text):
    """Parse JSON from LLM response."""
    if text.startswith("```"):
        text = text.split("```")[1].removeprefix("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def normalize_query(user_query, trace=None):
    """Extract filters from user query using GPT, including subject and level."""
    catalog, topic_to_lesson, lesson_sections = _ensure_catalog()

    prompt = f"""Identify lesson metadata and user profile from this question. Return ONLY JSON, omit unknown fields.
Available: lesson_number: {catalog['lesson_number']}, topic: {catalog['topic']}, section_name: {catalog['section_name']}
Subject values: FR (Faster Reading), FM (Faster Math)
Level values: Oral, Letter, Word, Sentence, Story
Question: "{user_query}"
Return: {{"lesson_number": "X", "section_name": "Y", "subject": "FR/FM", "level": "Oral/Letter/Word/Sentence/Story"}}"""

    # span = log_span(trace, "normalize_query", input_data={"question": user_query})

    text = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    ).choices[0].message.content.strip()

    filters = _parse_json(text)
    logger.info(f"Filters extracted: {filters}")

    # if span:
    #     span.end(output=filters)

    # Validate subject
    valid_subjects = ["FR", "FM"]
    if "subject" in filters and filters["subject"] not in valid_subjects:
        logger.warning(f"Invalid subject: {filters['subject']}")
        filters.pop("subject")

    # Validate level
    valid_levels = ["Oral", "Letter", "Word", "Sentence", "Story"]
    if "level" in filters and filters["level"] not in valid_levels:
        logger.warning(f"Invalid level: {filters['level']}")
        filters.pop("level")

    # Resolve topic → lesson_number
    if "topic" in filters and "lesson_number" not in filters:
        topic_val = filters.pop("topic")
        if topic_val in topic_to_lesson:
            filters["lesson_number"] = topic_to_lesson[topic_val]

    # Fuzzy-match section_name
    if "section_name" in filters:
        raw = filters["section_name"]
        lesson_num = filters.get("lesson_number")
        pool = lesson_sections.get(lesson_num, catalog["section_name"]) if lesson_num else catalog["section_name"]

        match = get_close_matches(raw, pool, n=1, cutoff=FUZZY_CUTOFF)
        if match:
            filters["section_name"] = match[0]
        else:
            filters.pop("section_name")

    # Validate filters exist in catalog (except subject and level)
    validated = {f: v for f, v in filters.items() if f not in ["subject", "level"] and f in catalog and v in catalog[f]}

    # Keep subject and level in validated
    if "subject" in filters:
        validated["subject"] = filters["subject"]
    if "level" in filters:
        validated["level"] = filters["level"]

    return validated


def search_lessons(user_query, trace=None):
    """Search lessons by query."""
    _ensure_catalog()

    filters = normalize_query(user_query, trace)

    # span = log_span(trace, "search_vector", input_data={"query": user_query, "filters": filters})
    embedding = _get_embedding(user_query)

    rows = _supabase_post("rpc/search_lessons_vector", {
        "query_embedding": embedding,
        "match_count": MATCH_COUNT,
        "filter_lesson_number": filters.get("lesson_number"),
        "filter_section_name": filters.get("section_name"),
    })

    # if span:
    #     span.end(output={"result_count": len(rows)})

    return {
        "documents": [[r["content"] for r in rows]],
        "metadatas": [[{
            "lesson_number": r["lesson_number"],
            "section_name": r["section_name"],
            "duration": r.get("duration", "not specified"),
        } for r in rows]],
        "distances": [[r.get("similarity", 0) for r in rows]],
        "ids": [[r["id"] for r in rows]],
    }, filters


def generate_answer(results, user_query):
    """Generate AI answer from results."""
    if not results["documents"][0]:
        raise ValueError("No lessons to answer from")

    retrieved = "\n\n".join([
        f"[Lesson {m['lesson_number']}, {m['section_name']}]:\n{doc}"
        for doc, m in zip(results["documents"][0], results["metadatas"][0])
    ])

    return client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Content:\n{retrieved}\n\nQuestion: {user_query}"}
        ],
        max_tokens=500
    ).choices[0].message.content


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _ensure_catalog()
        logger.info("✓ Server ready")
    except Exception as e:
        logger.error(f"✗ Startup failed: {e}")
        raise
    yield


app = FastAPI(title="Lesson Query API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Question(BaseModel):
    question: str
    wa_id: str


@app.get("/")
async def root():
    return {"name": "Lesson Query API", "version": "1.0.0", "docs": "/docs"}


@app.get("/health")
async def health():
    catalog, _, _ = _ensure_catalog()
    return {"status": "ok", "lessons": len(catalog["lesson_number"])}


@app.get("/catalog")
async def get_catalog():
    catalog, _, _ = _ensure_catalog()
    return {
        "lesson_numbers": catalog["lesson_number"],
        "topics": catalog["topic"],
        "sections": catalog["section_name"],
        "objectives": catalog["objective"]
    }


@app.post("/query")
async def query(req: Question):
    """Search lessons."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question required")

    # trace = log_trace("query", metadata={"wa_id": req.wa_id}, input_data={"question": req.question})
    trace = None

    try:
        logger.info(f"Query from wa_id: {req.wa_id}")

        cached = check_wa_id_24h(req.wa_id)
        is_new_session = cached == -1

        results, filters = search_lessons(req.question, trace)
        subject = filters.get("subject")
        level = filters.get("level")
        if subject or level:
            update_wa_id_access(req.wa_id, subject=subject, level=level)

        response = {**results, "filters_applied": filters, "wa_id": req.wa_id, "is_new_session": is_new_session, "cached": cached if not is_new_session else None, "extracted": {"subject": subject, "level": level}}

        # if trace:
        #     trace.end(output=response, metadata={"status": "success", "subject": subject, "level": level})

        return response
    except Exception as e:
        logger.error(f"Query error (wa_id: {req.wa_id}): {e}")
        # if trace:
        #     trace.end(output={"error": str(e)}, metadata={"status": "error"})
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/answer")
async def answer(req: Question):
    """Get AI answer."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question required")

    # trace = log_trace("answer", metadata={"wa_id": req.wa_id}, input_data={"question": req.question})
    trace = None

    try:
        logger.info(f"Answer request from wa_id: {req.wa_id}")

        cached = check_wa_id_24h(req.wa_id)
        is_new_session = cached == -1

        results, filters = search_lessons(req.question, trace)
        if not results["ids"][0]:
            raise HTTPException(status_code=404, detail="No lessons found")

        subject = filters.get("subject")
        level = filters.get("level")
        if subject or level:
            update_wa_id_access(req.wa_id, subject=subject, level=level)

        # span = log_span(trace, "generate_answer", input_data={"question": req.question, "context_count": len(results["documents"][0])})
        answer_text = generate_answer(results, req.question)
        # if span:
        #     span.end(output={"answer": answer_text})

        response = {
            "answer": answer_text,
            "question": req.question,
            "wa_id": req.wa_id,
            "is_new_session": is_new_session,
            "cached": cached if not is_new_session else None,
            "extracted": {"subject": subject, "level": level},
            "sources": [
                {
                    "lesson_number": m["lesson_number"],
                    "section_name": m["section_name"],
                    "duration": m["duration"]
                }
                for m in results["metadatas"][0]
            ]
        }

        # if trace:
        #     trace.end(output=response, metadata={"status": "success", "subject": subject, "level": level, "source_count": len(results["metadatas"][0])})

        return response
    except HTTPException as he:
        # if trace:
        #     trace.end(output={"error": str(he.detail)}, metadata={"status": "error"})
        raise
    except Exception as e:
        logger.error(f"Answer error (wa_id: {req.wa_id}): {e}")
        # if trace:
        #     trace.end(output={"error": str(e)}, metadata={"status": "error"})
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
