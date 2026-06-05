"""
Pydantic data models for request/response validation.

These models define the structure of data exchanged with the API and internally
within the LLM pipeline. Pydantic automatically validates incoming JSON and
ensures all required fields are present with correct types.
"""

from pydantic import BaseModel
from typing import Optional


class QueryNormalization(BaseModel):
    """
    Output of the LLM when extracting metadata from a user query.

    The LLM reads the user's question and extracts structured fields like
    the subject (FR/FM), level (Oral/Letter/Word/Sentence/Story), lesson number, etc.
    """
    subject: Optional[str] = None
    level: Optional[str] = None
    lesson_number: Optional[str] = None
    topic: Optional[str] = None
    section_name: Optional[str] = None
    is_command_only: bool = False


class SubjectLevel(BaseModel):
    """
    Output of Pass 1 normalization (subject + level extraction only).
    """
    subject: Optional[str] = None
    level: Optional[str] = None
    is_command_only: bool = False


class LessonMetadata(BaseModel):
    """
    Output of Pass 2 normalization (lesson metadata extraction from filtered catalog).
    """
    lesson_number: Optional[str] = None
    topic: Optional[str] = None
    section_name: Optional[str] = None


class Question(BaseModel):
    """
    Input to the /answer endpoint.

    The WhatsApp bot sends the user's question and their unique WhatsApp ID.
    """
    question: str
    wa_id: str
