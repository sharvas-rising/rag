"""
Lesson Query API - FastAPI Server

This application provides a RAG (Retrieval-Augmented Generation) system for
teachers in Sub-Saharan Africa to query lesson content and get AI-generated answers.

Architecture:
- FastAPI endpoints: /answer (search + generate), /health
- Database: Supabase (vector embeddings, lesson content)
- LLM: OpenAI GPT-4o-mini for both embeddings and answer generation
"""

# ============================================================================
# IMPORTS
# ============================================================================

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
from typing import Optional


class QueryNormalization(BaseModel):
    subject: Optional[str] = None
    level: Optional[str] = None
    lesson_number: Optional[str] = None
    topic: Optional[str] = None
    section_name: Optional[str] = None
    clean_query: str

# ============================================================================
# CONFIGURATION
# ============================================================================

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API keys from environment
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Validate required keys
if not OPENAI_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY not found in .env")

# LLM configuration
EMBEDDING_MODEL = "text-embedding-3-small"  # For generating query embeddings
LLM_MODEL = "gpt-4o-mini"  # For generating answers
MATCH_COUNT = 7  # Number of lessons to retrieve for context
FUZZY_CUTOFF = 0.3  # Threshold for fuzzy-matching lesson names

# System prompt for the AI
SYSTEM_PROMPT = (
    "You are a friendly teaching buddy helping teachers in Sub-Saharan Africa 🌍💚. "
    "Their English may be basic, so always:\n"
    "- Use very simple, clear English (short sentences, common words)\n"
    "- Be warm and encouraging, like a supportive peer — not a formal expert\n"
    "- Add relevant emojis to make it feel friendly and easy to read ✨\n"
    "- Keep answers short and to the point\n"
    "Answer only using the lesson content provided."
)

# ============================================================================
# EXTERNAL API CLIENTS
# ============================================================================

# Initialize OpenAI client for embeddings and LLM calls
client = OpenAI(api_key=OPENAI_KEY)

# Initialize Supabase session with authentication headers
session = requests.Session()
session.headers.update({
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
})

# ============================================================================
# CATALOG MANAGEMENT (Cached)
# ============================================================================

# Cache for lesson catalog to avoid repeated database queries
_catalog = None


def _supabase_get(endpoint, params=None):
    """
    Make a GET request to Supabase REST API.

    Args:
        endpoint: API endpoint (e.g., 'lessons_chunks')
        params: Query parameters

    Returns:
        JSON response from Supabase
    """
    resp = session.get(
        f"{SUPABASE_URL}/rest/v1/{endpoint}",
        params=params,
        headers={"Accept": "application/json"},
        timeout=10
    )
    resp.raise_for_status()
    return resp.json()


def _supabase_post(endpoint, json_data):
    """
    Make a POST request to Supabase REST API (for RPC calls).

    Args:
        endpoint: API endpoint (e.g., 'rpc/search_lessons_vector')
        json_data: Request body

    Returns:
        JSON response from Supabase
    """
    resp = session.post(
        f"{SUPABASE_URL}/rest/v1/{endpoint}",
        json=json_data,
        headers={"Content-Type": "application/json"},
        timeout=10
    )
    resp.raise_for_status()
    return resp.json()


def check_wa_id_24h(wa_id):
    """
    Check if a user (wa_id) has an active session within the past 24 hours.

    Returns:
        Dictionary with {subject, level} if active session exists
        -1 if no active session (new user or session expired)
    """
    try:
        rows = _supabase_get("user_access_log", params={
            "wa_id": f"eq.{wa_id}",
            "select": "subject,level,access_time"
        })

        if rows:
            record = rows[0]
            access_time = datetime.fromisoformat(record["access_time"].replace("Z", "+00:00"))
            # Check if record is within 24 hours
            if datetime.now(access_time.tzinfo) - access_time < timedelta(hours=24):
                return {"subject": record["subject"], "level": record["level"]}

        return -1
    except Exception as e:
        logger.error(f"Error checking wa_id {wa_id}: {e}")
        return -1


def update_wa_id_access(wa_id, subject=None, level=None):
    """
    Update user access log with latest session info (upsert).

    Since wa_id is the primary key:
    - Creates a new record if user is new
    - Updates the existing record if user returns (merges on duplicate wa_id)

    The Prefer: resolution=merge-duplicates header enables upsert semantics
    instead of failing on duplicate wa_id.

    Args:
        wa_id: User's WhatsApp ID (phone number)
        subject: Learning subject (FR=Faster Reading, FM=Faster Math)
        level: Learning level (Oral, Letter, Word, Sentence, Story)
    """
    try:
        resp = session.post(
            f"{SUPABASE_URL}/rest/v1/user_access_log",
            json={
                "wa_id": wa_id,
                "access_time": datetime.utcnow().isoformat(),
                "subject": subject,
                "level": level,
            },
            headers={
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates",
            },
            timeout=10
        )
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Error updating access log for {wa_id}: {e}")


def _ensure_catalog():
    """
    Load and cache the lesson catalog from Supabase.

    Returns:
        Dict with lesson_number, topic, section_name, objective, subject, level
    """
    global _catalog

    if _catalog is not None:
        return _catalog

    rows = _supabase_get("lessons_chunks", params={
        "select": "lesson_number,topic,section_name,objective,subject,level"
    })

    catalog = {field: set() for field in ["lesson_number", "topic", "section_name", "objective", "subject", "level"]}
    for row in rows:
        for field in catalog:
            if row.get(field):
                catalog[field].add(row[field])

    _catalog = {k: sorted(v) if v else [] for k, v in catalog.items()}
    logger.info(f"Catalog loaded: {len(_catalog.get('lesson_number', []))} lessons")
    return _catalog


# ============================================================================
# LLM OPERATIONS
# ============================================================================

@lru_cache(maxsize=128)
def _get_embedding(text):
    """
    Generate and cache embedding for text using OpenAI.

    Uses LRU cache to avoid re-embedding the same text.

    Args:
        text: Text to embed

    Returns:
        Embedding vector
    """
    return client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    ).data[0].embedding




def normalize_query(user_query):
    """
    Extract structured metadata from user query using GPT and return cleaned query.

    Uses LLM to identify:
    - lesson_number: Which lesson is relevant
    - section_name: Which section of the lesson
    - subject: Learning subject (FR or FM)
    - level: Learning level (Oral, Letter, Word, Sentence, Story)
    - clean_query: Query rewritten with metadata terms removed

    Args:
        user_query: User's natural language question

    Returns:
        Tuple of (filters_dict, clean_query_str)
    """
    catalog = _ensure_catalog()

    normalization_prompt = f"""
    You are a query normalization system.

    Your task is to extract lesson metadata from a user's query and produce a cleaned query.

    Rules:
    - Extract only metadata that is explicitly mentioned or clearly matches one of the available values.
    - Never guess or infer missing metadata.
    - Match values only from the provided catalogs.
    - Remove extracted metadata terms from the query when creating clean_query.
    - Preserve the educational or pedagogical intent of the query.
    - If removing metadata would leave the query empty, use the original query.
    - If a field cannot be extracted, return null for that field.
    - Subject can be called by the user as Faster Reading (FR) or Faster Math (FM)

    Available values:

    Subjects:
    {catalog['subject']}

    Levels:
    {catalog['level']}

    Lesson Numbers:
    {catalog['lesson_number']}

    Topics:
    {catalog['topic']}

    Section Names:
    {catalog['section_name']}
    """
    user_prompt = f""" User query: {user_query} """

    response = client.responses.parse(
    model="gpt-4o-mini",
    instructions=normalization_prompt,
    input=user_prompt,
    text_format=QueryNormalization,
    )

    return response.output_parsed


def search_lessons(user_query, subject_level, wa_id):
    """
    Search Supabase for lessons matching the query.

    Process:
    1. Normalize query to extract metadata (subject, level, lesson number)
    2. Merge with cached session subject/level as fallback
    3. Generate embedding for the query
    4. Perform vector similarity search in Supabase
    5. Update user session if subject/level changed from cache

    Args:
        user_query: User's question
        subject_level: Cached session dict or -1 if new user
        wa_id: WhatsApp user ID for tracking

    Returns:
        Tuple of (results, filters_dict)
        - results: Dict with documents, metadata, distances, ids
        - filters_dict: Dict of extracted metadata
    """
    _ensure_catalog()

    result = normalize_query(user_query)
    clean_query = result.clean_query or user_query

    logger.info(f"Original query: {user_query} Clean query: {clean_query}  Normalization: {result}")

    session_data = subject_level if isinstance(subject_level, dict) else {}
    subject = result.subject or session_data.get("subject")
    level = result.level or session_data.get("level")
    lesson_number = result.lesson_number
    section_name = result.section_name

    subject_changed = subject and subject != session_data.get("subject")
    level_changed = level and level != session_data.get("level")
    if subject_changed or level_changed:
        update_wa_id_access(wa_id, subject=subject, level=level)

    embedding = _get_embedding(clean_query)

    rows = _supabase_post("rpc/search_lessons_vector", {
        "query_embedding": embedding,
        "match_count": MATCH_COUNT,
        "filter_subject": subject,
        "filter_level": level,
        "filter_lesson_number": lesson_number,
        "filter_section_name": section_name,
    })

    if not rows or not isinstance(rows, list):
        logger.warning(f"No results from Supabase: {type(rows)}")
        rows = []

    filters_dict = {
        "subject": subject,
        "level": level,
        "lesson_number": lesson_number,
        "section_name": section_name,
    }

    return {
        "documents": [[r["content"] for r in rows]],
        "metadatas": [[{
            "lesson_number": r["lesson_number"],
            "section_name": r["section_name"],
            "duration": r.get("duration", "not specified"),
        } for r in rows]],
        "distances": [[r.get("similarity", 0) for r in rows]],
        "ids": [[r["id"] for r in rows]],
    }, filters_dict


def generate_answer(results, user_query):
    """
    Generate AI answer using retrieved lesson content as context.

    Uses GPT to synthesize an answer based on:
    - Retrieved lesson content from search_lessons()
    - User's original question
    - System prompt (friendly, simple English for African teachers)

    Args:
        results: Search results from search_lessons()
        user_query: User's original question

    Returns:
        Generated answer text
    """
    if not results["documents"][0]:
        raise ValueError("No lessons to answer from")

    # Format retrieved lessons as context
    retrieved = "\n\n".join([
        f"[Lesson {m['lesson_number']}, {m['section_name']}]:\n{doc}"
        for doc, m in zip(results["documents"][0], results["metadatas"][0])
    ])

    # Generate answer using GPT
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Content:\n{retrieved}\n\nQuestion: {user_query}"}
        ],
        max_tokens=500
    )

    return response.choices[0].message.content


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle handler.
    Startup: Load and cache lesson catalog
    Shutdown: Cleanup resources
    """
    try:
        # Pre-load catalog into memory on startup to avoid delays on first request
        _ensure_catalog()
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    yield
    # Shutdown cleanup happens here if needed


app = FastAPI(
    title="Lesson Query API",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================================
# DATA MODELS
# ============================================================================

class Question(BaseModel):
    """Request model for /answer endpoint."""
    question: str
    wa_id: str


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check and API info."""
    return {
        "name": "Lesson Query API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    catalog, _, _ = _ensure_catalog()
    return {
        "status": "ok",
        "lessons": len(catalog["lesson_number"])
    }


@app.post("/answer")
async def answer(req: Question):
    """
    Search for lessons and generate an AI answer.

    Combines /query functionality with LLM-based answer generation.
    Returns synthesized answer based on retrieved lesson content.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question required")

    try:
        subject_level = check_wa_id_24h(req.wa_id)
        is_new_session = subject_level == -1

        results, filters = search_lessons(req.question, subject_level, req.wa_id)
        if not results["ids"][0]:
            return -1
        answer_text = generate_answer(results, req.question)

        return {
            "answer": answer_text,
            "question": req.question,
            "wa_id": req.wa_id,
            "is_new_session": is_new_session,
            "cached": subject_level if not is_new_session else None,
            "extracted": {"subject": filters.get("subject"), "level": filters.get("level")},
            "sources": [
                {
                    "lesson_number": m["lesson_number"],
                    "section_name": m["section_name"],
                    "duration": m["duration"]
                }
                for m in results["metadatas"][0]
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Answer error (wa_id: {req.wa_id}): {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
