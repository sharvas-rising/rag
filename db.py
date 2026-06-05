"""
Database operations and session management via Supabase.

This module handles all interactions with the Supabase REST API, including:
- Loading the lesson catalog (cached in memory)
- Tracking user sessions (subject/level)
- Searching for lesson chunks by embedding
"""

import logging
from datetime import datetime, timedelta
from config import session, SUPABASE_URL, langfuse
from langfuse import observe

logger = logging.getLogger(__name__)

# Cache for lesson catalog (loaded once at startup, then reused)
_catalog = None


def _supabase_get(endpoint, params=None):
    """
    Make a GET request to the Supabase REST API.

    Args:
      endpoint: API endpoint path (e.g., 'lessons_chunks' or 'user_access_log')
      params: Query parameters (e.g., {'wa_id': 'eq.1234567890'})

    Returns:
      JSON response from Supabase (usually a list of rows)
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
    Make a POST request to the Supabase REST API (for RPC calls and inserts).

    Args:
      endpoint: API endpoint path (e.g., 'rpc/search_lessons_vector')
      json_data: JSON body to send

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


def _ensure_catalog():
    """
    Load and cache the lesson catalog from Supabase.

    The catalog contains all unique values for subject, level, lesson_number, topic, etc.
    It's fetched once at startup and reused. This avoids repeated database queries.

    Also builds a topic_to_lesson map for fast topic → lesson_number lookups.

    Returns:
      Dict with keys: subject, level, lesson_number, topic, section_name, topic_to_lesson
    """
    global _catalog

    if _catalog is not None:
        return _catalog

    # Fetch all unique lesson metadata from the database
    rows = _supabase_get("lessons_chunks", params={
        "select": "lesson_number,topic,section_name,objective,subject,level"
    })

    # Build sets of unique values for each field
    catalog = {field: set() for field in ["lesson_number", "topic", "section_name", "objective", "subject", "level"]}
    topic_to_lesson = {}

    for row in rows:
        # Add each field value to its set
        for field in catalog:
            if row.get(field):
                catalog[field].add(row[field])

        # Build topic → lesson mapping for quick lookups
        # e.g., topic "Actions" maps to lessons ["9", "10"]
        topic = row.get("topic")
        lesson = row.get("lesson_number")
        if topic and lesson:
            if topic not in topic_to_lesson:
                topic_to_lesson[topic] = []
            if lesson not in topic_to_lesson[topic]:
                topic_to_lesson[topic].append(lesson)

    # Convert sets to sorted lists for easier reading
    _catalog = {k: sorted(v) if v else [] for k, v in catalog.items()}
    _catalog["topic_to_lesson"] = topic_to_lesson
    _catalog["rows"] = rows  # Store raw rows for filtering by subject+level later

    logger.info(f"Catalog loaded: {len(_catalog.get('lesson_number', []))} unique lessons, "
                f"{len(_catalog.get('subject', []))} subjects, {len(_catalog.get('level', []))} levels")
    return _catalog


def get_filtered_catalog(subject=None, level=None):
    """
    Get a filtered view of the catalog for a specific subject and level.

    This is used in Pass 2 of normalization — after we know the subject+level,
    we show the LLM only the topics/lessons/sections that exist for that combination.
    This prevents hallucinations (e.g. extracting "Vocabulary" if it's not in the filtered set).

    Args:
      subject: Subject code (e.g., "FR" or "FM") or None
      level: Level name (e.g., "Oral", "Letter") or None

    Returns:
      Dict with same structure as full catalog, but filtered to only matching rows
    """
    full_catalog = _ensure_catalog()
    rows = full_catalog.get("rows", [])

    # Filter rows to only those matching the subject and level
    if subject:
        rows = [r for r in rows if r.get("subject") == subject]
    if level:
        rows = [r for r in rows if r.get("level") == level]

    # Rebuild unique sets + topic_to_lesson mapping from filtered rows
    filtered = {field: set() for field in ["lesson_number", "topic", "section_name", "subject", "level"]}
    topic_to_lesson = {}

    for row in rows:
        for field in filtered:
            if row.get(field):
                filtered[field].add(row[field])

        topic = row.get("topic")
        lesson = row.get("lesson_number")
        if topic and lesson:
            if topic not in topic_to_lesson:
                topic_to_lesson[topic] = []
            if lesson not in topic_to_lesson[topic]:
                topic_to_lesson[topic].append(lesson)

    # Convert sets to sorted lists
    result = {k: sorted(v) if v else [] for k, v in filtered.items()}
    result["topic_to_lesson"] = topic_to_lesson
    return result


def check_wa_id_24h(wa_id):
    """
    Check if a user (WhatsApp ID) has an active session within the past 24 hours.

    If they do, we return their cached subject/level. If not (new user or session expired),
    return -1 as a sentinel value.

    Args:
      wa_id: User's WhatsApp ID (phone number as string)

    Returns:
      Dict with {subject, level} if active session exists
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
            if datetime.now(access_time.tzinfo) - access_time < timedelta(hours=240):
                return {"subject": record["subject"], "level": record["level"]}

        return -1
    except Exception as e:
        logger.error(f"Error checking wa_id {wa_id}: {e}")
        return -1


@observe()
def update_wa_id_access(wa_id, subject=None, level=None):
    """
    Update or insert a user session record in the access log.

    This records when the user last accessed the API and what subject/level they're using.
    Uses an upsert (merge on duplicate) so it creates a new record for new users and
    updates existing records when they return.

    Only updates if BOTH subject and level are provided (not None).

    Args:
      wa_id: User's WhatsApp ID (phone number as string)
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
                "Prefer": "resolution=merge-duplicates",  # Enables upsert on duplicate wa_id
            },
            timeout=10
        )
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Error updating access log for {wa_id}: {e}")
