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
from app.utils.log_utils import async_save_meta_log

# main.py 에 정의된 의존성 주입 함수 임포트 -> dependencies.py 로 변경
from app.dependencies import get_mcp_service, get_inference_service

api_router = APIRouter()

# 로거 설정
logger = logging.getLogger(__name__)

# --- Placeholder Translation Function -> LLM Translation ---
async def _translate_if_needed(
    text: str,
    inference_service: InferenceService, # InferenceService 인스턴스를 직접 받음
    target_language: str = "en"
) -> str:
    """입력 텍스트의 언어를 감지하고 필요한 경우 대상 언어로 번역합니다.
    한국어(ko)만 특별 처리하며, 다른 모든 언어는 영어로 처리합니다."""
    
    try:
        # 한국어인지 아닌지만 판단
        source_language = classify_language(text)
    except Exception as e:
        logger.warning(f"Language detection failed: {e}. Assuming not Korean.")
        source_language = "en" # 기본값은 영어로 설정
        
    # 번역이 필요 없는 경우 (한국어→한국어 또는 비한국어→영어)
    if (source_language == "ko" and target_language == "ko") or (source_language != "ko" and target_language == "en"):
        return text
        
    try:
        # LLM을 이용한 번역
        if not inference_service or not inference_service.model_loaded:
             logger.error("Inference service not available for translation.")
             return text # 번역 서비스 없으면 원문 반환

        prompt = ""
        stop_sequences = ["\\n", "Korean text:", "English translation:", "영어 텍스트:", "한국어 번역:"]
        
        # 한국어 → 영어 번역
        if source_language == "ko" and target_language == "en":
            prompt = f"You are a translation engine. Translate the following Korean text to English. Output ONLY the English translation, without any introductory phrases, explanations, or quotation marks.\\n\\nKorean text: '{text}'\\nEnglish translation:"
        
        # 영어/기타 → 한국어 번역 (source_language가 무엇이든 한국어가 아니면 영어로 간주)
        elif target_language == "ko":
            prompt = f"You are a translation engine. Translate the following English text to Korean. Output ONLY the Korean translation, without any introductory phrases, explanations, or quotation marks.\\n\\n영어 텍스트: '{text}'\\n한국어 번역:"
        
        # 번역 생성
        translated_text = await inference_service.generate(
            prompt=prompt,
            max_tokens=len(text) * 3, # 번역 결과 길이를 고려하여 토큰 수 설정
            temperature=0.2, # 번역은 낮은 온도로
            stop=stop_sequences
        )
        # 번역 결과 후처리 (예: 불필요한 공백 제거)
        return translated_text.strip()
    except Exception as e:
        logger.error(f"Translation failed: {e}", exc_info=True)
        return text # 번역 실패 시 원문 반환

# --- Chat Endpoint Models --- 
class ChatRequest(BaseModel):
    text: str

class ThoughtAction(BaseModel):
    thought: str
    action: Dict[str, Any] = {}
    observation: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    thoughts_and_actions: Optional[List[ThoughtAction]] = None
    full_response: Optional[str] = None
    error: Optional[str] = None
    log_session_id: Optional[str] = None
    log_path: Optional[str] = None
    detected_language: Optional[str] = None

# --- Health Endpoint Model (from status.py) --- 
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    model_exists: bool
    mcp_servers: list
    version: str = "0.1.0" # api.py에는 없었으므로 추가
    
    # Pydantic v2 경고 해결 (model_ 필드 충돌 방지)
    model_config = {
        "protected_namespaces": ()
    }

# --- Chat Endpoint (from api.py) --- 
@api_router.post("/chat", response_model=ChatResponse, tags=["Chat"], response_model_exclude_none=True)
async def chat_endpoint(
    chat_request: ChatRequest,
    request: Request,
    inference_service: InferenceService = Depends(get_inference_service), # 의존성 주입 사용
    settings: Settings = Depends(get_settings) 
):
    """
    질문이나 명령을 받아 AI가 응답하는 단일 엔드포인트
    ReAct 패턴을 사용하여 필요시 도구를 활용하며 응답합니다.
    각 단계별 로그는 로그 디렉토리에 자세히 저장됩니다.
    """
    session_id = str(int(datetime.now().timestamp()))
    logger.info(f"POST /chat request received (Session ID: {session_id}), Text: {chat_request.text}")
    
    # --- 커스텀 언어 감지 사용 --- 
    original_text = chat_request.text
    lang_for_processing = classify_language(original_text)
    logger.info(f"[{session_id}] Language classified for processing: {lang_for_processing}")
    
    # --- Translate Input to English (if necessary) --- 
    prompt_for_llm = original_text
    if lang_for_processing == 'ko':
        prompt_for_llm = await _translate_if_needed(original_text, inference_service, target_language='en')

    # --- 요청 로그 기록 --- 
    log_dir = Path(settings.log_dir)
    
    # 클라이언트 IP 주소 가져오기
    client_host = request.client.host if request.client else "unknown"
    
    # X-Forwarded-For 또는 X-Real-IP 헤더가 있는 경우 (프록시 뒤에 있는 경우) 사용
    forwarded_for = request.headers.get("X-Forwarded-For")
    real_ip = request.headers.get("X-Real-IP")
    
    if forwarded_for:
        # X-Forwarded-For는 쉼표로 구분된 IP 목록일 수 있음 (가장 왼쪽이 원본 클라이언트)
        client_host = forwarded_for.split(",")[0].strip()
    elif real_ip:
        client_host = real_ip
        
    logger.debug(f"Client host for session {session_id}: {client_host}")
        
    request_log_data = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "event_type": "api_request",
        "endpoint": "/chat",
        "method": "POST",
        "client_host": client_host, 
        "request_body": chat_request.dict(),
        "language_for_processing": lang_for_processing,
        "prompt_sent_to_llm": prompt_for_llm
    }
    await async_save_meta_log(log_dir, session_id, request_log_data)
    
    log_file_path_str = str(log_dir / "api_logs" / session_id / "meta.json")
    
    try:
        # ReAct 패턴 실행 (inference_service 는 주입받은 것 사용)
        result = await inference_service.process_react_pattern(
            initial_prompt=prompt_for_llm,
            session_id=session_id,
            max_iterations=10,  # 최대 반복 횟수 제한
            log_dir=str(log_dir)  # 로그 디렉토리 명시적 지정
        )
        
        # 도구 정보 추가 (실제 사용된 도구에 대한 정보만 제공)
        # 사용하려고 시도한 도구 이름 추출
        used_tool_name = None
        used_server_name = None
        
        # result의 metadata에서 마지막 iteration 확인
        if result and "metadata" in result and "iterations" in result["metadata"]:
            iterations = result["metadata"]["iterations"]
            if iterations:
                last_iteration = iterations[-1]
                if "action" in last_iteration and "tool_name" in last_iteration["action"]:
                    full_tool_name = last_iteration["action"]["tool_name"]
                    
                    # tool_name에서 서버 이름과 도구 이름 분리 (서버.도구 형식)
                    if "." in full_tool_name:
                        used_server_name, used_tool_name = full_tool_name.split(".", 1)
                    else:
                        used_tool_name = full_tool_name
        
        # 실제 사용된 도구에 대한 정보만 제공
        if used_tool_name:
            logger.info(f"[{session_id}] 사용된 도구 감지: {used_tool_name} (서버: {used_server_name or '알 수 없음'})")
            
            mcp_service = get_mcp_service()
            tool_hint = ""
            
            try:
                if mcp_service:
                    running_servers = mcp_service.get_running_servers()
                    
                    # 서버 이름을 모르는 경우, 모든 실행 중인 서버에서 도구 찾기
                    server_to_check = [used_server_name] if used_server_name else running_servers
                    
                    for server_name in server_to_check:
                        if server_name not in running_servers:
                            continue
                            
                        # 해당 도구에 대한 상세 정보 가져오기
                        tool_details = mcp_service.get_tool_details(server_name, used_tool_name)
                        
                        if 'description' in tool_details:  # 도구 정보가 존재하는 경우
                            tool_hint = f"\n\nSPECIFIC TOOL INFORMATION:\n"
                            tool_hint += f"Tool: {server_name}.{used_tool_name}\n"
                            tool_hint += f"Description: {tool_details.get('description', 'No description')}\n"
                            
                            # 파라미터 정보 추가
                            parameters = tool_details.get('parameters', {})
                            if parameters:
                                tool_hint += f"Parameters:\n"
                                for param_name, param_info in parameters.items():
                                    required = "Required" if param_info.get('required', False) else "Optional"
                                    tool_hint += f"  - {param_name}: {param_info.get('description', 'No description')} ({required})\n"
                            
                            # 첫 번째 일치하는 도구 정보를 찾으면 종료
                            break
                    
            except Exception as e:
                logger.warning(f"[{session_id}] 도구 정보 가져오기 오류: {e}")
            
            # 도구 정보가 있는 경우에만 추가
            if tool_hint:
                prompt_for_llm = f"{prompt_for_llm}\n{tool_hint}"
                logger.debug(f"[{session_id}] 특정 도구({used_tool_name})에 대한 정보만 프롬프트에 추가")
        
        # 새로운 반환 형식 처리
        english_response_str = result.get("response", "")
        metadata = result.get("metadata", {})
        error_msg = None
        
        # 메타데이터에서 필요한 정보 추출
        iterations_data = metadata.get("iterations", [])
        
        # 생각과 행동 데이터 정리
        thoughts_data = []
        for iteration in iterations_data:
            thought = ""
            action = iteration.get("action", {})
            if action:
                # "thought" 필드 추출 (새 형식에 맞게)
                thought = action.get("thought", "")
            
            thoughts_data.append({
                "thought": thought,
                "action": action,
                "observation": iteration.get("observation", "")
            })
        
        # 전체 응답 생성 (이전 형식과 비슷하게)
        full_english_response_str = ""
        if hasattr(inference_service, '_format_full_response'):
            full_english_response_str = inference_service._format_full_response(thoughts_data)
        else:
            # 대체 방법으로 전체 응답 생성
            responses = []
            for iteration in iterations_data:
                response = iteration.get("response", "")
                observation = iteration.get("observation", "")
                if response:
                    responses.append(response)
                if observation:
                    responses.append(f"Observation: {observation}")
            full_english_response_str = "\n\n".join(responses)
        
        # 로그 레벨 확인
        is_debug = settings.log_level.upper() == "DEBUG"
        logger.debug(f"Log level check: {settings.log_level.upper()}, is_debug={is_debug}")
        
        response_obj_data = {
            "response": english_response_str,
            "error": None, 
            "log_session_id": session_id,
            "detected_language": lang_for_processing, 
            "log_path": log_file_path_str
        }
        
        if is_debug:
            thoughts_and_actions = []
            if thoughts_data:
                for item in thoughts_data:
                    action_data = item.get("action", {})
                    if action_data is None: action_data = {}
                    elif not isinstance(action_data, dict):
                        action_data = {"raw": str(action_data)}
                    
                    thoughts_and_actions.append(
                        ThoughtAction(
                            thought=item.get("thought", ""),
                            action=action_data,
                            observation=item.get("observation", "")
                        )
                    )

            response_obj_data["thoughts_and_actions"] = thoughts_and_actions
            response_obj_data["full_response"] = full_english_response_str
            response_obj_data["error"] = error_msg
        else: 
             # 비-디버그 시 error 필드는 최종 번역 후 설정
             pass

        response_log_data = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "event_type": "api_response",
            "endpoint": "/chat",
            "status_code": 200,
            "response_body": response_obj_data
        }
        await async_save_meta_log(log_dir, session_id, response_log_data)

        # --- Translate Output back to Original Language (if necessary) --- 
        final_response_for_user = english_response_str
        
        # 오류 메시지 처리 (번역 제거)
        if error_msg:
             final_response_for_user = error_msg
             logger.warning(f"[{session_id}] ReAct process returned an error.")
             # Update error field in response_obj_data for non-debug case
             if not is_debug:
                 response_obj_data["error"] = error_msg
        
        # Update the response field with the final response
        response_obj_data["response"] = final_response_for_user
        
        # Create the final response object
        response_obj = ChatResponse(**response_obj_data)
        
        return response_obj

    except Exception as e:
        logger.error(f"Error in /chat endpoint for session_id {session_id}: {e}", exc_info=True)
        # ... (오류 처리 로직은 log_file_path_str을 이미 사용 가능)
        lang_info = lang_for_processing if 'lang_for_processing' in locals() else 'unknown'

        error_response_obj = ChatResponse(
            response=f"An error occurred during processing (Session: {session_id}, Language: {lang_info}): {str(e)}",
            error=str(e),
            log_session_id=session_id,
            detected_language=lang_info,
            log_path=log_file_path_str # 미리 정의된 경로 사용
        )
        # ... (오류 로그 기록 및 반환)
        return JSONResponse(
             status_code=500,
             content=error_response_obj.dict(exclude_none=True)
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

    model_path_str = str(inference_service.model_path) if inference_service and inference_service.model_path else "Not configured"
    model_exists = inference_service.model_path.exists() if inference_service and inference_service.model_path else False
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