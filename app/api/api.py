import logging
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime
import traceback

from app.utils.lang_utils import classify_language

from app.services.mcp_service import MCPService
from app.core.config import settings
from app.services.inference_service import InferenceService
from app.utils.log_utils import async_save_meta_log, get_session_log_directory
from app.dependencies import get_mcp_service, get_inference_service

api_router = APIRouter()

logger = logging.getLogger(__name__)

def get_session_id(request: Request) -> str:
    """Generates or retrieves a session ID."""
    return str(int(datetime.now().timestamp() * 1000))

async def _translate_text(
    text: str, 
    target_language: str, 
    source_language: str, 
    inference_service: InferenceService
) -> Optional[str]:
    """Translates text between supported languages using the AI model."""
    
    # 1. Check if translation is necessary
    if source_language == target_language:
        logger.debug(f"Source and target languages are the same ({source_language}). No translation needed.")
        return text

    # 2. Define language names for the prompt
    language_map = {"ko": "Korean", "en": "English"}
    source_lang_name = language_map.get(source_language, source_language)
    target_lang_name = language_map.get(target_language, target_language)

    # 3. Construct the prompt
    prompt = (
        f"Translate the following {source_lang_name} text to {target_lang_name}. "
        f"Output ONLY the translated {target_lang_name} text, without any introductory phrases, explanations, or quotation marks."
        f"\n{source_lang_name} Text: {text}"
        f"\n{target_lang_name} Translation:"
    )
    messages = [{"role": "user", "content": prompt}]
        
    translated_text = await inference_service.model_service.generate_chat(
        messages=messages, 
        grammar=None
    )

    # 4. Clean the response
    cleaned_translation = translated_text.strip()
    if cleaned_translation:
        logger.debug(f"Translated {source_language} -> {target_language}: '{text}' -> '{cleaned_translation}'")
        return cleaned_translation
    else:
        logger.warning(f"Translation result is empty after cleaning ({source_language}->{target_language}). Original response: '{translated_text}'. Returning None.")
        return None

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
    detected_language: Optional[str] = None
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
    Handles language detection and translation (KO <-> EN).
    Logs the interaction details to a session-specific meta.json file.
    """
    session_id = chat_request.session_id or get_session_id(request)
    base_log_dir = Path(settings.log_dir)
    session_log_dir = get_session_log_directory(base_log_dir, session_id)
    log_file_path = session_log_dir / "meta.json"
    
    session_log_dir.mkdir(parents=True, exist_ok=True)

    # Variables to store final results
    final_response_for_user = "An unexpected error occurred."
    response_for_user_en = "An unexpected error occurred."
    error_detail_for_response: Optional[ErrorDetail] = None

    try:
        logger.debug(f"Received raw request body: {chat_request}")

        # --- Language Handling (Input) ---
        user_input = chat_request.text
        detected_lang = chat_request.language or classify_language(user_input)
        logger.info(f"Detected language: {detected_lang} for session {session_id}")
        
        prompt_for_llm = user_input
        if detected_lang == 'ko':
            prompt_for_llm = await _translate_text(user_input, 'en', 'ko', inference_service)

        # --- Initial Logging ---
        initial_meta = {
            "session_id": session_id,
            "start_time": datetime.now().isoformat(),
            "initial_request": user_input,
            "language_detected": detected_lang,
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
                response_for_user_en = f"Error processing your request: {error_detail_for_response.details}"
                await async_save_meta_log(
                    session_log_dir,
                    {"errors": [f"{error_detail_for_response.message}: {error_detail_for_response.details}"], "event_type": "react_error"},
                    session_id,
                    merge=True
                )
            else:
                # ReAct processing successful
                response_for_user_en = result.get("response", "No response generated.")
                error_detail_for_response = None # No error

        except Exception as react_exc:
            logger.error(f"Unhandled error during ReAct processing for session {session_id}: {react_exc}", exc_info=True)
            error_detail_for_response = ErrorDetail(
                code=500,
                message="Internal server error during ReAct processing",
                details=str(react_exc)
            )
            response_for_user_en = f"Sorry, an error occurred: {error_detail_for_response.details}"
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
        response_for_user_en = f"Sorry, an unexpected error occurred: {error_detail_for_response.details}"
        try:
            await async_save_meta_log(
                session_log_dir,
                {"errors": [f"Unhandled endpoint error: {str(e)}"], "event_type": "unhandled_endpoint_error"},
                session_id,
                merge=True
            )
        except Exception as log_err:
            logger.error(f"Failed to log unhandled endpoint error for session {session_id}: {log_err}")

    # --- Final Response Translation (if needed) ---
    final_response_for_user = response_for_user_en
    if detected_lang == 'ko':
        logger.info(f"Translating final EN response back to Korean for session {session_id}: {response_for_user_en[:100]}...")
        try:
            translated_response = await _translate_text(response_for_user_en, 'ko', 'en', inference_service)
            if translated_response:
                final_response_for_user = translated_response
                logger.info(f"Response translated EN -> KO for session {session_id}")
            else:
                logger.warning(f"EN -> KO translation failed for final response. Returning English for session {session_id}")
        except Exception as trans_e:
            logger.error(f"Error during final response translation for session {session_id}: {trans_e}", exc_info=True)

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

    # --- Return Response --- 
    return ChatResponse(
        response=final_response_for_user,
        error=error_detail_for_response, 
        log_session_id=session_id,
        log_path=str(log_file_path) if log_file_path.exists() else None,
        detected_language=detected_lang
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