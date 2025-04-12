import logging
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime
import traceback
import json

from app.utils.lang_utils import classify_language

from app.services.mcp_service import MCPService # get_mcp_service 제거
from app.core.config import Settings, get_settings

# InferenceService 임포트 (이제 파일이 존재해야 함)
from app.services.inference_service import InferenceService # get_inference_service 제거

# 로그 유틸리티 임포트
from app.utils.log_utils import async_save_meta_log, get_session_log_directory

# main.py 에 정의된 의존성 주입 함수 임포트 -> dependencies.py 로 변경
from app.dependencies import get_mcp_service, get_inference_service

api_router = APIRouter()

# 로거 설정
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def get_session_id(request: Request) -> str:
    """Generate or retrieve a session ID."""
    return str(int(datetime.now().timestamp() * 1000))

async def _translate_text(text: str, target_language: str, inference_service: InferenceService) -> Optional[str]:
    """Translates text using the inference service (LLM).
       Specifically handles KO -> EN translation.
       Returns original text if input is not KO or target is not EN.
    """
    source_language = classify_language(text)

    if not (source_language == 'ko' and target_language == 'en'):
        logger.debug(f"Translation not required for {source_language} -> {target_language}. Returning original text.")
        return text

    prompt = f"Translate the following Korean text to English. Output ONLY the English translation, without any introductory text or quotation marks.\nKorean Text: '{text}'\nEnglish Translation:"
    stop_sequences = ["\n", "Korean Text:"]
    messages = [{"role": "user", "content": prompt}] # Use chat format

    try:
        if not inference_service or not inference_service.model_loaded:
            logger.error("Inference service not available for KO->EN translation.")
            return None

        if not hasattr(inference_service, 'model_service') or not inference_service.model_service:
             logger.error("ModelService not found within InferenceService for KO->EN translation.")
             return None

        # Use generate_chat instead of generate
        translated_text = await inference_service.model_service.generate_chat(
            messages=messages, # Pass messages list
            max_tokens=len(text) * 3 + 50,
            temperature=0.2,
            stop=stop_sequences
        )

        cleaned_translation = translated_text.strip()
        if cleaned_translation.startswith("English Translation:"):
            cleaned_translation = cleaned_translation.replace("English Translation:", "").strip()

        logger.info(f"Translated KO -> EN: '{text[:50]}...' -> '{cleaned_translation[:50]}...'")
        return cleaned_translation

    except Exception as e:
        logger.error(f"KO -> EN translation error: {e}", exc_info=True)
        return None

async def _translate_en_to_ko(text: str, inference_service: InferenceService) -> Optional[str]:
    """Translates English text to Korean using the inference service (LLM)."""

    prompt = f"Translate the following English text to Korean. Output ONLY the Korean translation, without any introductory text or quotation marks.\nEnglish Text: '{text}'\nKorean Translation:"
    stop_sequences = ["\n", "English Text:"]
    messages = [{"role": "user", "content": prompt}] # Use chat format

    try:
        if not inference_service or not inference_service.model_loaded:
            logger.error("Inference service not available for EN->KO translation.")
            return None

        if not hasattr(inference_service, 'model_service') or not inference_service.model_service:
             logger.error("ModelService not found within InferenceService for EN->KO translation.")
             return None

        # Use generate_chat instead of generate
        translated_text = await inference_service.model_service.generate_chat(
            messages=messages, # Pass messages list
            max_tokens=len(text) * 4 + 50,
            temperature=0.2,
            stop=stop_sequences
        )

        cleaned_translation = translated_text.strip()
        if cleaned_translation.startswith("Korean Translation:"):
             cleaned_translation = cleaned_translation.replace("Korean Translation:", "").strip()

        logger.info(f"Translated EN -> KO: '{text[:50]}...' -> '{cleaned_translation[:50]}...'")
        return cleaned_translation

    except Exception as e:
        logger.error(f"EN -> KO translation error: {e}", exc_info=True)
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
    version: str = "0.1.0"

    # Pydantic v2 model_ config for protected namespaces if needed
    model_config = {
        "protected_namespaces": ()
    }

# --- API Endpoints ---
@api_router.post("/chat", response_model=ChatResponse, tags=["Chat"], response_model_exclude_none=True)
async def chat_endpoint(
    chat_request: ChatRequest,
    request: Request,
    inference_service: InferenceService = Depends(get_inference_service), # 의존성 주입 사용
    settings: Settings = Depends(get_settings)
):
    """
    Handle incoming chat requests, process them using the InferenceService with ReAct pattern,
    and return the final response.
    Handles language detection and translation (KO <-> EN).
    Logs the interaction details to a session-specific meta.json file.
    """
    session_id = chat_request.session_id or get_session_id(request)
    client_host = request.client.host if request.client else "unknown"
    user_input = chat_request.text

    # --- Logging Setup ---
    # Use the helper function to get the correct log directory
    log_dir = Path(settings.log_dir) # Base log directory from settings
    session_log_dir = get_session_log_directory(log_dir, session_id)
    log_file_path = session_log_dir / "meta.json"

    logger.info(f"Received chat request for session {session_id} from {client_host}")
    logger.debug(f"Session log directory: {session_log_dir}")

    # --- Language Detection and Translation ---
    detected_lang = classify_language(user_input)
    lang_for_processing = detected_lang
    prompt_for_llm = user_input # Default to original input
    if detected_lang == 'ko':
        logger.info(f"Korean input detected for session {session_id}. Attempting translation to English.")
        try:
            translated_prompt = await _translate_text(user_input, 'en', inference_service)
            if translated_prompt:
                prompt_for_llm = translated_prompt
                logger.info(f"Input successfully translated KO -> EN for session {session_id}.")
                lang_for_processing = 'en'
            else:
                logger.error(f"KO -> EN translation failed for session {session_id}. Using original Korean prompt.")
                lang_for_processing = 'ko' # Keep original language if translation fails
                # Ensure prompt_for_llm remains the original user_input here
                prompt_for_llm = user_input 
        except Exception as e:
            logger.error(f"Error during input translation call for session {session_id}: {e}", exc_info=True)
            lang_for_processing = 'ko' # Fallback to original language on error
            prompt_for_llm = user_input # Ensure original prompt is used on error
    else:
         lang_for_processing = detected_lang if detected_lang else 'en'
         logger.info(f"Input language '{detected_lang}' detected. Proceeding without translation for LLM.")
         prompt_for_llm = user_input # Explicitly set for non-KO cases

    # --- Log API Request --- 
    # Log the actual prompt being sent to the inference service
    logger.info(f"Prompt being sent to process_react_pattern for session {session_id}: '{prompt_for_llm[:100]}...'")
    
    model_path_info = "Unknown"
    model_params_info = {}
    if inference_service and inference_service.model_service:
        if inference_service.model_loaded:
             model_path_info = str(inference_service.model_service.model_path)
             model_params_info = inference_service.model_service.model_params
        else:
             model_path_info = str(inference_service.model_service.model_path or settings.model_path or "Not configured")
             model_params_info = inference_service.model_service.model_params
    else:
         logger.warning("InferenceService or ModelService not fully initialized, cannot log model path.")

    request_log_data = {
        "timestamp": datetime.now().isoformat(),
        "event_type": "api_request",
        "client_host": client_host,
        "request_body": chat_request.model_dump(exclude_none=True),
        "language_detected": detected_lang,
        "prompt_sent_to_llm": prompt_for_llm,
        "model": { # Add model info here
             "path": model_path_info,
             "params": model_params_info
        }
    }
    await async_save_meta_log(session_log_dir, session_id, request_log_data)

    # --- Process Request using InferenceService --- 
    try:
        react_result = await inference_service.process_react_pattern(
            initial_prompt=prompt_for_llm,
            session_id=session_id,
            session_log_dir=session_log_dir, # Pass correct log dir
            max_iterations=20
        )

        final_answer_llm = react_result.get("response", "Sorry, I could not generate a response.")

        # --- Translate response back if original was Korean ---
        final_answer_user = final_answer_llm
        if detected_lang == 'ko':
            logger.info(f"Original input was Korean. Attempting to translate EN response back to KO for session {session_id}.")
            try:
                translated_response = await _translate_en_to_ko(final_answer_llm, inference_service)
                if translated_response:
                    final_answer_user = translated_response
                    logger.info(f"Response successfully translated EN -> KO for session {session_id}.")
                else:
                    logger.error(f"EN -> KO translation failed for session {session_id}. Returning original English response.")
                    # final_answer_user remains final_answer_llm (English)
            except Exception as e:
                logger.error(f"Error during response translation call for session {session_id}: {e}", exc_info=True)
                # final_answer_user remains final_answer_llm (English)
        # If input was not KO, final_answer_user remains the English final_answer_llm

        # --- Log API Response --- 
        response_log_data = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "api_response",
            "status_code": 200,
            "response_body": { # Structure matches ChatResponse
                 "response": final_answer_user,
            }
        }
        await async_save_meta_log(session_log_dir, session_id, response_log_data)

        # --- Return Response --- 
        return ChatResponse(
            response=final_answer_user,
            log_session_id=session_id,
            log_path=str(log_file_path)
        )

    except HTTPException as http_exc:
        error_log_data = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "api_error",
            "error_details": {
                "type": "HTTPException",
                "status_code": http_exc.status_code,
                "detail": http_exc.detail
            }
        }
        await async_save_meta_log(session_log_dir, session_id, error_log_data)
        raise http_exc

    except Exception as e:
        logger.error(f"Unhandled error during chat processing for session {session_id}: {e}", exc_info=True)
        error_details_trace = traceback.format_exc()
        error_log_data = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "api_error",
            "error_details": {
                "type": "UnhandledException",
                "message": str(e),
                "traceback": error_details_trace
            }
        }
        await async_save_meta_log(session_log_dir, session_id, error_log_data)

        return JSONResponse(
            status_code=500,
            content=ChatResponse(
                response="An unexpected error occurred. Please try again later.",
                error=ErrorDetail(code=500, message="Internal Server Error"),
                log_session_id=session_id,
                log_path=str(log_file_path)
            ).model_dump(exclude_none=True)
        )

# --- Health Endpoint (from status.py) --- 
@api_router.get("/health", response_model=HealthResponse, tags=["Status"])
async def health_check(
    request: Request,
    mcp_service: MCPService = Depends(get_mcp_service), # 의존성 주입 사용
    inference_service: InferenceService = Depends(get_inference_service), # 의존성 주입 사용
    # settings: Settings = Depends(get_settings) # 제거 (필요 시 서비스 통해 접근)
):
    """
    애플리케이션의 상태 (모델 로딩, MCP 서버 연결 등)를 확인합니다.
    """
    # 클라이언트 IP 주소 가져오기
    client_host = request.client.host if request.client else "unknown"
    forwarded_for = request.headers.get("X-Forwarded-For")
    real_ip = request.headers.get("X-Real-IP")
    
    if forwarded_for:
        client_host = forwarded_for.split(",")[0].strip()
    elif real_ip:
        client_host = real_ip
        
    logger.info(f"GET /health request received from {client_host}")

    # 서비스 인스턴스는 Depends로 주입받음 (request.app.state 제거)
    # mcp_service: MCPService = request.app.state.mcp_service
    # inference_service: InferenceService = request.app.state.inference_service

    # Check model status using model_loaded attribute
    model_path_str = "Unknown"
    model_exists = False
    if inference_service and inference_service.model_service and inference_service.model_service.model_path:
         model_path_obj = Path(inference_service.model_service.model_path)
         model_path_str = str(model_path_obj)
         model_exists = model_path_obj.exists()
    elif settings.model_path: # Fallback to settings if service not fully ready
         model_path_obj = Path(settings.model_path)
         model_path_str = str(model_path_obj)
         model_exists = model_path_obj.exists()
    
    model_loaded_status = inference_service.model_loaded if inference_service else False

    # MCP 서버 목록 가져오기 (get_running_servers 사용)
    running_servers = []
    if mcp_service:
        try:
            running_servers = mcp_service.get_running_servers() # get_active_servers -> get_running_servers
        except Exception as e:
             logger.error(f"Error getting running MCP servers during health check: {e}", exc_info=True)
             
    return HealthResponse(
        status="OK" if mcp_service and inference_service and model_loaded_status else "PARTIALLY_UNAVAILABLE", # 모델 로드 상태 반영
        model_loaded=model_loaded_status,
        model_path=model_path_str,
        model_exists=model_exists,
        mcp_servers=running_servers, # 실행 중인 서버 목록 반환
        version="0.1.0" 
    ) 