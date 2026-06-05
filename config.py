"""
Configuration and initialization for the Lesson Query API.

This module loads environment variables, validates required keys, and initializes
external API clients (OpenAI, Supabase) so they can be imported throughout the app.
"""

import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
from langfuse import Langfuse

# --- Load environment variables from .env file ---
load_dotenv()

# --- Load and validate API keys ---
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not OPENAI_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY not found in .env")

# --- LLM model and embedding settings ---
EMBEDDING_MODEL = "text-embedding-3-small"  # Used to convert user queries into vectors
LLM_MODEL = "gpt-4o-mini"  # Used for query normalization and answer generation
MATCH_COUNT = 7  # Number of lesson chunks to retrieve per search

# --- Initialize OpenAI client ---
# Used by normalize_query() and generate_answer() to call GPT-4o-mini
client = OpenAI(api_key=OPENAI_KEY)

# --- Initialize Supabase HTTP session ---
# Used by _supabase_get() and _supabase_post() for all database operations
session = requests.Session()
session.headers.update({
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
})

# --- Initialize LangFuse for tracing (debugging tool) ---
# Captures all function calls and their inputs/outputs for debugging
try:
    langfuse = Langfuse()
except Exception as e:
    print(f"Warning: LangFuse initialization failed ({e}). Tracing disabled.")
    langfuse = None
