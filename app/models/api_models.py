from typing import Optional
from pydantic import BaseModel

class ChatRequest(BaseModel):
    text: str
    session_id: Optional[str] = None
    # Remove language field
    # language: Optional[str] = None # Optional: 'ko', 'en', etc. If not provided, will be detected.

class ChatResponse(BaseModel):
    response: str
    error: Optional[ErrorDetail] = None
    log_session_id: str
    log_path: Optional[str] = None
    # Remove detected_language field
    # detected_language: Optional[str] = None

class HealthResponse(BaseModel):
    status: str 