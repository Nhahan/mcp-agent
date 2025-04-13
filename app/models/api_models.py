from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union


class ToolRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    

class ToolResponse(BaseModel):
    content: List[Dict[str, Any]]


class ChatRequest(BaseModel):
    """Chat request from the user."""
    message: str
    conversation_id: Optional[str] = None
    initial_prompt: Optional[str] = None
    
    
class ChatResponse(BaseModel):
    """Response to a chat request."""
    answer: str
    conversation_id: str