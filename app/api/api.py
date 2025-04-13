import logging
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, AsyncGenerator
from datetime import datetime
import json

from app.core.config import settings
from app.services.inference_service import InferenceService
from app.services.mcp_service import MCPService
from app.utils.log_utils import async_save_meta_log, get_session_log_directory
from app.dependencies import get_mcp_service, get_inference_service

api_router = APIRouter()

logger = logging.getLogger(__name__)

async def _stream_json_data(data_generator):
    """Helper function to properly format JSON data for Server-Sent Events (SSE)"""
    if isinstance(data_generator, list):
        # If data_generator is a list, convert it to an async generator
        async def list_to_generator():
            for item in data_generator:
                yield item
        data_generator = list_to_generator()
    
    try:
        async for data in data_generator:
            if isinstance(data, str):
                # If data is already a JSON string, use it directly
                json_str = data
            else:
                # Otherwise convert to JSON
                json_str = json.dumps(data)
            # Format according to SSE spec: each message prefixed with "data: " and ending with "\n\n"
            yield f"data: {json_str}\n\n"
    except Exception as e:
        logger.error(f"Error in _stream_json_data: {e}", exc_info=True)
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

def get_session_id(request: Request) -> str:
    """Generates or retrieves a session ID."""
    return str(int(datetime.now().timestamp() * 1000))

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    text: str
    session_id: Optional[str] = None # Allow client to provide session ID
    language: Optional[str] = None # Optional language hint from client

class ErrorDetail(BaseModel):
    code: int
    message: str
    details: Optional[Any] = None

class IterationLog(BaseModel):
    iteration: int
    timestamp: str
    prompt: Optional[str] = None # Prompt for this iteration (can be large)
    response: Optional[str] = None # LLM response (can be large)
    action: Optional[Dict[str, Any]] = None # Parsed action (tool call or final answer)
    observation: Optional[Any] = None # Result of tool call
    error: Optional[str] = None # Error during this iteration

class ChatResponse(BaseModel):
    response: str
    error: Optional[ErrorDetail] = None
    log_session_id: str
    log_path: Optional[str] = None # Path to the meta.json log file
    # thoughts_and_actions field removed as it will be in meta.json
    full_response: Optional[Dict[str, Any]] = None # Keep for potential full debug output if needed

# --- Health Endpoint Model --- 
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    model_exists: bool
    mcp_servers: list
    version: str = settings.api_version # Use from settings

    # Pydantic v2 model_ config for protected namespaces if needed
    model_config = {
        "protected_namespaces": ()
    }

# --- API Endpoints ---
@api_router.post("/chat", response_model=ChatResponse, tags=["Chat"], response_model_exclude_none=True)
async def chat_endpoint(
    chat_request: ChatRequest,
    request: Request,
    inference_service: InferenceService = Depends(get_inference_service),
):
    """
    Handle incoming chat requests, process them using the InferenceService with ReAct pattern,
    and return the final response.
    Logs the interaction details to a session-specific meta.json file.
    """
    session_id = chat_request.session_id or get_session_id(request)
    base_log_dir = Path(settings.log_dir)
    session_log_dir = get_session_log_directory(base_log_dir, session_id)
    log_file_path = session_log_dir / "meta.json"
    
    session_log_dir.mkdir(parents=True, exist_ok=True)

    # Variables to store final results
    final_response_for_user = "An unexpected error occurred."
    error_detail_for_response: Optional[ErrorDetail] = None

    try:
        logger.debug(f"Received raw request body: {chat_request}")

        # --- Remove Language Handling --- 
        user_input = chat_request.text
        
        prompt_for_llm = user_input 
        initial_meta = {
            "session_id": session_id,
            "start_time": datetime.now().isoformat(),
            "initial_request": user_input,
            "model_info": { 
                "name": Path(settings.model_path).name if settings.model_path else "unknown",
                "path": settings.model_path,
                "parameters": { "n_ctx": settings.n_ctx, "gpu_layers": settings.gpu_layers }
            },
            "iterations": [],
            "errors": [],
            "request_timestamp": datetime.now().isoformat(),
        }
        await async_save_meta_log(session_log_dir, {"event_type": "api_request", **initial_meta}, session_id)

        # --- ReAct Pattern Processing ---
        try:
            result = await inference_service.process_react_pattern(
                initial_prompt=prompt_for_llm,
                session_id=session_id,
                session_log_dir=session_log_dir,
            )
            
            if "error" in result:
                # ReAct processing error
                logger.error(f"ReAct process failed for session {session_id}: {result['error']}")
                error_detail_for_response = ErrorDetail(
                    code=500,
                    message="ReAct processing failed",
                    details=result.get("error")
                )
                final_response_for_user = f"Error processing your request: {error_detail_for_response.details}"
                await async_save_meta_log(
                    session_log_dir,
                    {"errors": [f"{error_detail_for_response.message}: {error_detail_for_response.details}"], "event_type": "react_error"},
                    session_id,
                    merge=True
                )
            else:
                # ReAct processing successful
                final_response_for_user = result.get("response", "No response generated.")
                error_detail_for_response = None # No error

        except Exception as react_exc:
            logger.error(f"Unhandled error during ReAct processing for session {session_id}: {react_exc}", exc_info=True)
            error_detail_for_response = ErrorDetail(
                code=500,
                message="Internal server error during ReAct processing",
                details=str(react_exc)
            )
            final_response_for_user = f"Sorry, an error occurred: {error_detail_for_response.details}"
            await async_save_meta_log(
                session_log_dir,
                {"errors": [f"Unhandled ReAct error: {str(react_exc)}"], "event_type": "unhandled_react_error"},
                session_id,
                merge=True
            )
            
    except Exception as e:
        logger.error(f"Unhandled error in chat endpoint for session {session_id}: {e}", exc_info=True)
        error_detail_for_response = ErrorDetail(
            code=500,
            message="Internal server error",
            details=str(e)
        )
        final_response_for_user = f"Sorry, an unexpected error occurred: {error_detail_for_response.details}"
        try:
            await async_save_meta_log(
                session_log_dir,
                {"errors": [f"Unhandled endpoint error: {str(e)}"], "event_type": "unhandled_endpoint_error"},
                session_id,
                merge=True
            )
        except Exception as log_err:
            logger.error(f"Failed to log unhandled endpoint error for session {session_id}: {log_err}")

    # --- Final Logging --- 
    logger.debug(f"Final response content being logged for session {session_id}: {final_response_for_user[:100]}...")
    final_event_data = {
        "event_type": "api_response",
        "response_body": final_response_for_user, 
        "timestamp": datetime.now().isoformat()
    }
    try:
        await async_save_meta_log(session_log_dir, final_event_data, session_id, merge=True)
    except Exception as log_err:
         logger.error(f"Failed to save final api_response log for session {session_id}: {log_err}")

    # --- Return Response (remove detected_language) --- 
    log_file_path = session_log_dir / "meta.json"
    return ChatResponse(
        response=final_response_for_user,
        error=error_detail_for_response, 
        log_session_id=session_id,
        log_path=str(log_file_path) if log_file_path.exists() else None,
        # detected_language=detected_lang # Removed
    )

@api_router.get("/health", response_model=HealthResponse, tags=["Status"])
async def health_check(
    request: Request,
    mcp_service: MCPService = Depends(get_mcp_service),
    inference_service: InferenceService = Depends(get_inference_service),
):
    """
    Check the health/status of the application, including model loading status
    and available MCP servers.
    """
    if not inference_service or not inference_service.model_service:
        raise HTTPException(status_code=500, detail="InferenceService or ModelService not initialized")
    
    model_path = inference_service.model_service.model_path
    model_exists = Path(model_path).exists() if model_path else False
    model_loaded = inference_service.model_service.model_loaded if hasattr(inference_service.model_service, 'model_loaded') else False
    
    try:
        mcp_servers = mcp_service.list_servers() if mcp_service else []
    except Exception as e:
        logger.error(f"Error listing MCP servers in health check: {e}", exc_info=True)
        mcp_servers = []
    
    return HealthResponse(
        status="ok" if model_loaded else "model_not_loaded",
        model_loaded=model_loaded,
        model_path=model_path,
        model_exists=model_exists,
        mcp_servers=mcp_servers,
        version=settings.api_version # Use settings directly
    ) 

@api_router.post("/stream", tags=["Chat"])
async def stream_chat(
    chat_request: ChatRequest,
    request: Request = None,
    inference_service: InferenceService = Depends(get_inference_service),
):
    """
    Stream the AI reasoning process and final answer using Server-Sent Events via POST.
    Shows thinking responses during reasoning and only shows the final answer at the end.
    """
    # Extract text and session_id from the request body
    text = chat_request.text
    session_id = chat_request.session_id

    if not session_id:
        session_id = str(int(datetime.now().timestamp() * 1000))
    
    base_log_dir = Path(settings.log_dir)
    session_log_dir = get_session_log_directory(base_log_dir, session_id)
    session_log_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate the input text is not empty
    if not text.strip():
        return StreamingResponse(
            _stream_json_data([{"error": "Input text cannot be empty", "type": "error"}]),
            media_type="text/event-stream"
        )
    
    # Log the incoming request
    logger.info(f"[Stream] Starting processing for {session_id}: {text[:50]}...")
    
    async def event_generator() -> AsyncGenerator[str, None]:
        iterations_log = [] # Store detailed iteration logs
        try:
            prompt_for_llm = text
            initial_meta = {
                "session_id": session_id,
                "start_time": datetime.now().isoformat(),
                "initial_request": text,
                "model_info": { 
                     "name": Path(settings.model_path).name if settings.model_path else "unknown",
                     "path": settings.model_path,
                     "parameters": { "n_ctx": settings.n_ctx, "gpu_layers": settings.gpu_layers }
                },
                "iterations": [],
                "errors": [],
                "request_timestamp": datetime.now().isoformat(),
                "event_type": "api_request"
            }
            await async_save_meta_log(session_log_dir, initial_meta, session_id)

            # Stream the ReAct process
            async for step in inference_service.stream_react_pattern(
                initial_prompt=prompt_for_llm,
                session_id=session_id,
                session_log_dir=session_log_dir
            ):
                # Store iteration log from the step if available
                if step.get("iteration_log"): 
                    iterations_log.append(step["iteration_log"])
                
                # Determine step type
                if "step_type" in step:
                    step_type = step["step_type"]
                elif "thought" in step:
                    step_type = "thinking"
                elif "response" in step and "error" not in step:
                    step_type = "final_response"
                elif "error" in step:
                    step_type = "error"
                else:
                    # Log unknown step type for debugging
                    logger.warning(f"[Stream] Received unknown step type from generator for {session_id}: {step}")
                    step_type = "unknown"
                
                event_data = {}
                
                # Process based on step_type
                if step_type == "thinking":
                    # Thinking steps
                    thought_for_user = step.get("thought", "")
                    event_data = {"type": "thinking", "content": thought_for_user}
                    
                    # Include other potential fields
                    if "tool" in step: event_data["tool"] = step["tool"]
                    if "observation" in step: event_data["observation"] = step["observation"]
                    if "error" in step: event_data["error"] = step["error"]
                    if "feedback_provided" in step: event_data["feedback_provided"] = step["feedback_provided"]
                    
                elif step_type == "final_response":
                    # Final response
                    final_answer_for_user = step.get("response", "")                    
                    event_data = {"type": "final", "content": final_answer_for_user}
                    
                    # Log final response details
                    final_log_event = {
                        "event_type": "api_streaming_complete",
                        "final_response": final_answer_for_user,
                        "iterations": iterations_log,
                        "timestamp": datetime.now().isoformat()
                    }
                    await async_save_meta_log(session_log_dir, final_log_event, session_id, merge=True)
                    
                elif step_type == "error" or step_type == "error_response":
                    # Error processing
                    error_msg = step.get("error", "Unknown error")
                    error_content = step.get("response", f"Error: {error_msg}")
                    event_data = {"type": "error", "content": error_content, "error": error_msg}
                    
                else:
                    # Fallback for other types
                    event_data = {"type": step_type, "data": step}
                
                # Yield the event data
                yield json.dumps(event_data)

        except Exception as e:
            logger.error(f"Error during stream generation for session {session_id}: {e}", exc_info=True)
            error_data = {"type": "error", "content": f"Server error during streaming: {str(e)}"}
            yield json.dumps(error_data)
            # Log the final error
            final_error_log = {
                "event_type": "api_streaming_error",
                "error_details": str(e),
                "timestamp": datetime.now().isoformat()
            }
            await async_save_meta_log(session_log_dir, final_error_log, session_id, merge=True)
            
    return StreamingResponse(
        _stream_json_data(event_generator()),
        media_type="text/event-stream"
    ) 