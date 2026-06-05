"""
Lesson Query API - FastAPI Server for African Teachers

This is a RAG (Retrieval-Augmented Generation) system that helps teachers in
Sub-Saharan Africa find lesson content and get AI-generated answers to their questions.

Architecture:
  - FastAPI endpoints: /answer (search + generate), /health
  - Database: Supabase (vector embeddings, lesson content)
  - LLM: OpenAI GPT-4o-mini for both embeddings and answer generation
  - Session tracking: remembers user's subject/level preference for 24 hours
"""

import logging
from functools import lru_cache
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import client, EMBEDDING_MODEL, LLM_MODEL, MATCH_COUNT
from models import QueryNormalization, Question, SubjectLevel, LessonMetadata
from prompts import SYSTEM_PROMPT, build_subject_level_prompt, build_lesson_metadata_prompt
from db import _ensure_catalog, check_wa_id_24h, update_wa_id_access, _supabase_post, get_filtered_catalog
from langfuse import observe

# --- Configure logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@lru_cache(maxsize=128)
def _get_embedding(text):
    """
    Convert text into a vector (embedding) using OpenAI's embedding model.

    The embedding captures the semantic meaning of the text so Supabase can find
    similar lesson chunks using vector similarity search. We cache results to avoid
    re-embedding the same text multiple times.

    Args:
      text: Text to convert into a vector

    Returns:
      List of floats representing the vector
    """
    return client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    ).data[0].embedding


@observe()
def _extract_subject_level(user_query, cached_subject=None, cached_level=None):
    """
    Pass 1 of two-pass normalization: extract subject and level only.

    Uses the full catalog (all subjects and levels) since we don't know the
    context yet. After this pass, we merge with session cache so we always
    have at least subject+level for the database search.

    Args:
      user_query: The user's question
      cached_subject: Subject from user's cached session (if any)
      cached_level: Level from user's cached session (if any)

    Returns:
      SubjectLevel object with subject, level, is_command_only
    """
    catalog = _ensure_catalog()
    prompt = build_subject_level_prompt(catalog)
    user_prompt = f"User query: {user_query}"

    response = client.responses.parse(
        model=LLM_MODEL,
        instructions=prompt,
        input=user_prompt,
        text_format=SubjectLevel,
    )

    result = response.output_parsed
    # Merge with session cache
    result.subject = result.subject or cached_subject
    result.level = result.level or cached_level
    return result


@observe()
def _extract_lesson_metadata(user_query, filtered_catalog):
    """
    Pass 2 of two-pass normalization: extract lesson metadata from a filtered catalog.

    This is only called if subject+level are known (from pass 1 + cache).
    The filtered catalog contains ONLY rows matching that subject+level,
    so any value the LLM returns is guaranteed to be in the database.

    Args:
      user_query: The user's question
      filtered_catalog: Catalog filtered to subject+level (from get_filtered_catalog)

    Returns:
      LessonMetadata object with lesson_number, topic, section_name
    """
    prompt = build_lesson_metadata_prompt(filtered_catalog)
    user_prompt = f"User query: {user_query}"

    response = client.responses.parse(
        model=LLM_MODEL,
        instructions=prompt,
        input=user_prompt,
        text_format=LessonMetadata,
    )

    return response.output_parsed


def normalize_query(user_query, cached_subject=None, cached_level=None):
    """
    Two-pass LLM extraction of query metadata.

    Pass 1: Extract subject and level from full catalog, merge with session cache.
    Pass 2: If subject+level are known, filter catalog and extract lesson metadata.

    This prevents hallucinations — Pass 2 LLM only sees values that actually exist.

    Args:
      user_query: The user's question
      cached_subject: Subject from cached session (if any)
      cached_level: Level from cached session (if any)

    Returns:
      QueryNormalization object with all metadata, or sentinel for command-only queries
    """
    # --- Pass 1: Extract subject and level ---
    subject_level = _extract_subject_level(user_query, cached_subject, cached_level)

    # Early exit if this was a session-change command (e.g. "change to FR oral")
    if subject_level.is_command_only:
        return QueryNormalization(
            subject=subject_level.subject,
            level=subject_level.level,
            is_command_only=True,
        )

    subject = subject_level.subject
    level = subject_level.level

    # --- Pass 2: Extract lesson metadata from filtered catalog ---
    if subject and level:
        filtered_catalog = get_filtered_catalog(subject=subject, level=level)
        lesson_meta = _extract_lesson_metadata(user_query, filtered_catalog)
    else:
        # No subject/level, so no filtered catalog available
        lesson_meta = LessonMetadata()

    return QueryNormalization(
        subject=subject,
        level=level,
        lesson_number=lesson_meta.lesson_number,
        topic=lesson_meta.topic,
        section_name=lesson_meta.section_name,
        is_command_only=False,
    )


@observe()
def search_lessons(user_query, cached_subject_level, wa_id):
    """
    Find lesson chunks that best answer the user's question.

    Steps:
      1. Use LLM to extract metadata (subject, level, lesson, topic) from the query
      2. Fill in any missing subject/level from the user's cached 24-hour session
      3. If a topic was mentioned, look up its lesson number from the catalog
      4. Save the session if subject/level changed
      5. Search Supabase using one of two paths (precise or broad)

    Returns:
      (results, filters_dict) where results is lesson chunks or None if command-only
      or (None, filters_dict) if the query was a session-change command
    """
    _ensure_catalog()

    # --- Step 1: Ask the LLM to extract structured metadata from the raw query ---
    # Pass the cached subject/level so Pass 1 of normalization can use them
    session_data = cached_subject_level if isinstance(cached_subject_level, dict) else {}
    result = normalize_query(user_query, cached_subject=session_data.get("subject"), cached_level=session_data.get("level"))

    subject = result.subject
    level = result.level
    lesson_number = result.lesson_number

    logger.info("Normalization — Cached subject: %s | New subject: %s | Cached Level: %s | New level: %s | lesson: %s | topic: %s | command_only: %s",
                session_data.get("subject"), subject, session_data.get("level"), level, lesson_number, result.topic, result.is_command_only)

    # --- Step 3: Resolve topic → lesson number using the catalog lookup ---
    if result.topic and not lesson_number:
        topic_to_lesson = _ensure_catalog().get("topic_to_lesson", {})
        lessons = topic_to_lesson.get(result.topic)
        logger.info(f"Topic lookup: '{result.topic}' → {lessons}")
        if lessons:
            lesson_number = lessons[0]

    # --- Step 4: Save updated subject/level to the session if they changed ---
    # Only saves when BOTH are known — avoids overwriting a good value with None
    subject_changed = subject and subject != session_data.get("subject")
    level_changed = level and level != session_data.get("level")
    if (subject_changed or level_changed) and subject and level:
        update_wa_id_access(wa_id, subject=subject, level=level)

    filters_dict = {"subject": subject, "level": level, "lesson_number": lesson_number, "section_name": None}

    # If the message was purely a session change (e.g. "change to FR oral"), stop here.
    # The /answer endpoint will send a confirmation message instead.
    if result.is_command_only:
        return None, filters_dict

    # --- Step 5: Search Supabase ---
    # Convert the user query into a vector so Supabase can find the most
    # semantically similar lesson chunks.
    embedding = _get_embedding(user_query)

    search_path = None
    if subject and level and lesson_number:
        # PATH A — Precise: all three filters known, search inside one lesson only
        search_path = "A"
        rows = _supabase_post("rpc/search_lessons_vector", {
            "query_embedding": embedding,
            "match_count": MATCH_COUNT,
            "filter_subject": subject,
            "filter_level": level,
            "filter_lesson_number": lesson_number,
            "filter_section_name": None,
        })

    elif subject and level:
        # PATH B — Broad: lesson unknown, search across all lessons for this subject/level
        search_path = "B"
        rows = _supabase_post("rpc/search_lessons_vector", {
            "query_embedding": embedding,
            "match_count": MATCH_COUNT,
            "filter_subject": subject,
            "filter_level": level,
            "filter_lesson_number": None,
            "filter_section_name": None,
        })

    else:
        # Not enough context — subject and/or level missing
        search_path = "skipped"
        rows = []

    if not rows or not isinstance(rows, list):
        rows = []

    # Add search_path to filters_dict so it's visible in LangFuse traces
    filters_dict["search_path"] = search_path

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


@observe()
def generate_answer(results, user_query):
    """
    Generate an AI answer from retrieved lesson chunks.

    Uses GPT-4o-mini to synthesize a friendly, simple answer based on the
    retrieved lesson content. The system prompt tells it to use simple English
    and be encouraging for African teachers with basic English proficiency.

    Args:
      results: Dict with 'documents' and 'metadatas' from search_lessons()
      user_query: The user's original question

    Returns:
      String answer ready to send to the user
    """
    if not results["documents"][0]:
        raise ValueError("No lessons to answer from")

    # Format retrieved lessons as context for the LLM
    retrieved = "\n\n".join([
        f"[Lesson {m['lesson_number']}, {m['section_name']}]:\n{doc}"
        for doc, m in zip(results["documents"][0], results["metadatas"][0])
    ])

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Content:\n{retrieved}\n\nQuestion: {user_query}"}
        ],
        max_tokens=500
    )

    return response.choices[0].message.content


def _build_response(answer_text, req, is_new_session, cached_subject_level, filters, results=None):
    """
    Build the JSON response object sent back to the user.

    This helper avoids repeating the same dict structure across all the different
    answer paths (command-only, no results, normal question).

    Args:
      answer_text: The message to send the user
      req: The incoming request (contains question and wa_id)
      is_new_session: True if this user has no cached subject/level from before
      cached_subject_level: Dict of {subject, level} or -1 if new user
      filters: Dict of extracted metadata {subject, level, lesson_number}
      results: Search results dict or None if command-only

    Returns:
      Dict ready to be serialized to JSON
    """
    return {
        "answer": answer_text,
        "question": req.question,
        "wa_id": req.wa_id,
        "is_new_session": is_new_session,
        "cached": cached_subject_level if not is_new_session else None,
        "extracted": {"subject": filters.get("subject"), "level": filters.get("level")},
        "sources": [
            {"lesson_number": m["lesson_number"], "section_name": m["section_name"], "duration": m["duration"]}
            for m in results["metadatas"][0]
        ] if results else []
    }


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle handler.

    Startup: Load the lesson catalog into memory to avoid delays on first request
    Shutdown: (cleanup would go here if needed)
    """
    try:
        _ensure_catalog()
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    yield


app = FastAPI(
    title="Lesson Query API",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS so the WhatsApp bot can call us from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check and API info endpoint."""
    return {
        "name": "Lesson Query API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """Health check endpoint with lesson count."""
    catalog = _ensure_catalog()
    return {
        "status": "ok",
        "lessons": len(catalog.get("lesson_number", []))
    }


@app.post("/answer")
@observe()
async def answer(req: Question):
    """
    Answer a user's question using lesson content from the database.

    This is the main API endpoint. It handles three scenarios:
    1. User is changing their subject/level (no question asked)
    2. No matching lessons found for the user's question
    3. Normal case: find matching lessons and generate an AI answer
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question required")

    try:
        # --- Step 1: Check if this user has a cached session ---
        # If they asked about "FR oral" before, we remember it for 24 hours.
        cached_subject_level = check_wa_id_24h(req.wa_id)
        is_new_session = cached_subject_level == -1

        # --- Step 2: Search for matching lesson chunks ---
        # This also handles extracting subject/level/lesson/topic from the query.
        # Returns None if this was just a session-change command ("change to FR oral").
        results, filters = search_lessons(req.question, cached_subject_level, req.wa_id)
        subject = filters.get("subject")
        level = filters.get("level")

        # --- Step 3a: User was just changing their session ---
        # They said something like "change to FR letter level" with no actual question.
        # Confirm and ask them what they want to know.
        if results is None:
            parts = []
            if subject: parts.append(f"📚 Subject: {subject}")
            if level:   parts.append(f"🎯 Level: {level}")
            msg = f"✅ Updated! {' | '.join(parts)}\n\nWhat would you like to know? 😊" if parts else "✅ Got it! What would you like to know? 😊"
            return _build_response(msg, req, is_new_session, cached_subject_level, filters)

        # --- Step 3b: No matching lessons found ---
        # Could be because subject/level is missing, or no lessons match their question.
        if not results["ids"][0]:
            if not subject or not level:
                # They haven't told us their subject/level, so we can't search
                msg = "I need more info to help you! 📚\n\nPlease tell me:\n1️⃣ Your subject (FR or FM)\n2️⃣ Your level (Oral, Letter, Word, Sentence, or Story)\n\nThen ask your question! 😊"
            else:
                # They have subject/level, but no matching lessons
                msg = f"Sorry, I couldn't find lessons for {subject} at {level} level. 😕\n\nTry a different question or change your subject/level! 🔄"
            return _build_response(msg, req, is_new_session, cached_subject_level, filters)

        # --- Step 4: Generate the AI answer ---
        # Use the retrieved lesson chunks as context for the LLM.
        answer_text = generate_answer(results, req.question)

        # --- Step 5: Prepend status line ---
        # Show the user which subject/level/lesson we used, so they can see the context.
        status_parts = []
        if subject:                      status_parts.append(f"📚 Subject: {subject}")
        if level:                        status_parts.append(f"🎯 Level: {level}")
        if filters.get("lesson_number"): status_parts.append(f"📖 Lesson: {filters['lesson_number']}")
        if status_parts:
            answer_text = " | ".join(status_parts) + "\n\n" + answer_text

        return _build_response(answer_text, req, is_new_session, cached_subject_level, filters, results)

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
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
