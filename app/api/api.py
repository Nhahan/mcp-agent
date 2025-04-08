import logging
import os # status.py에서 가져옴
from pathlib import Path # status.py에서 가져옴
from fastapi import APIRouter, Depends, HTTPException # HTTPException은 api.py에서 가져옴
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime

# 서비스 임포트 - 양쪽 파일에서 공통적으로 사용
from app.services.inference_service import InferenceService, get_inference_service
from app.services.mcp_service import MCPService, get_mcp_service # status.py에서 가져옴
from app.core.config import Settings, get_settings

api_router = APIRouter()

# 로거 설정
logger = logging.getLogger(__name__)

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

# --- Health Endpoint (from status.py, path adjusted) --- 
@api_router.get("/health", response_model=HealthResponse, tags=["Health"])
async def get_health(
    inference_service: InferenceService = Depends(get_inference_service),
    mcp_service: MCPService = Depends(get_mcp_service)
):
    """시스템 상태를 반환하는 엔드포인트"""
    model_path = inference_service.model_path
    model_exists = model_path.exists()
    model_loaded = inference_service.model is not None
    
    # MCP 서버 목록 가져오기
    available_servers = mcp_service.get_available_server_names()
    running_servers = mcp_service.get_running_servers()
    
    mcp_status = [
        {"name": name, "running": name in running_servers}
        for name in available_servers
    ]
    
    logger.info(f"Health check: model_loaded={model_loaded}, model_exists={model_exists}")
    
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        model_path=str(model_path),
        model_exists=model_exists,
        mcp_servers=mcp_status
        # version은 모델 기본값 사용
    )

# --- Chat Endpoint (from api.py) --- 
@api_router.post("/chat", response_model=ChatResponse, tags=["Chat"], response_model_exclude_none=True)
async def chat_endpoint(
    request: ChatRequest,
    inference_service: InferenceService = Depends(get_inference_service),
    settings: Settings = Depends(get_settings) # Settings 주입
):
    """
    질문이나 명령을 받아 AI가 응답하는 단일 엔드포인트
    ReAct 패턴을 사용하여 필요시 도구를 활용하며 응답합니다.
    각 단계별 로그는 로그 디렉토리에 자세히 저장됩니다.
    """
    # 세션 ID 생성 (타임스탬프 기반)
    session_id = str(int(datetime.now().timestamp()))
    logger.info(f"POST /chat request received (Session ID: {session_id}), Text: {request.text[:50]}...")
    
    try:
        # ReAct 패턴을 사용하여 응답 생성 (session_id 전달)
        response_str, thoughts_data, full_response_str, error_msg = await inference_service.process_react_pattern(
            initial_prompt=request.text, 
            session_id=session_id
        )
        
        # 로그 레벨 확인
        is_debug = settings.log_level.upper() == "DEBUG"
        logger.debug(f"Log level check: {settings.log_level.upper()}, is_debug={is_debug}")

        if is_debug:
            # DEBUG 레벨일 경우: 상세 응답 반환
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

            log_path = f"logs/react_logs/{session_id}"
            
            logger.debug(f"Returning detailed response for DEBUG level. Session ID: {session_id}")
            return ChatResponse(
                response=response_str,
                thoughts_and_actions=thoughts_and_actions,
                full_response=full_response_str,
                error=error_msg,
                log_session_id=session_id,
                log_path=log_path
            )
        else:
            # DEBUG 레벨이 아닐 경우: response 필드만 반환
            logger.debug(f"Returning simplified response (response only) for non-DEBUG level. Session ID: {session_id}")
            # ChatResponse 모델을 사용하되, 필요한 필드만 채움
            return ChatResponse(
                response=response_str,
                log_session_id=session_id
            )
            
    except Exception as e:
        logger.error(f"Error in /chat endpoint for session_id {session_id}: {e}", exc_info=True)
        # 오류 발생 시에도 session_id를 포함하여 반환 시도
        # 로그 레벨과 관계없이 오류 시에는 상세 정보 반환
        return ChatResponse(
            response=f"처리 중 오류가 발생했습니다: {str(e)}",
            error=str(e),
            log_session_id=session_id
        ) 