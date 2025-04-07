import logging
from fastapi import APIRouter
from fastapi import HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from app.api.v1.endpoints import status
from app.services.inference_service import InferenceService, get_inference_service

api_router = APIRouter()

# Add logger for this module
logger = logging.getLogger(__name__)

# 채팅 요청 모델
class ChatRequest(BaseModel):
    text: str

class ThoughtAction(BaseModel):
    thought: str
    action: Dict[str, Any] = {}
    observation: str

class ChatResponse(BaseModel):
    response: str
    thoughts_and_actions: Optional[List[ThoughtAction]] = None
    full_response: Optional[str] = None
    error: Optional[str] = None
    log_session_id: Optional[str] = None
    log_path: Optional[str] = None

# 상태 확인 엔드포인트 포함
api_router.include_router(status.router, prefix="/status", tags=["Status"])

# 단일 채팅 엔드포인트
@api_router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(
    request: ChatRequest,
    inference_service: InferenceService = Depends(get_inference_service)
):
    """
    질문이나 명령을 받아 AI가 응답하는 단일 엔드포인트
    ReAct 패턴을 사용하여 필요시 도구를 활용하며 응답합니다.
    각 단계별 로그는 로그 디렉토리에 자세히 저장됩니다.
    """
    logger.info(f"Received request for /chat endpoint with text: {request.text[:50]}...")
    try:
        # ReAct 패턴을 사용하여 응답 생성
        logger.debug("Calling inference_service.process_react_pattern...")
        result = await inference_service.process_react_pattern(request.text)
        logger.debug("Returned from inference_service.process_react_pattern")
        
        # ThoughtAction 객체로 변환
        thoughts_and_actions = []
        if "thoughts_and_actions" in result:
            for item in result["thoughts_and_actions"]:
                # 새로운 JSON 구조에 맞게 변환
                action_data = item.get("action", {})
                if not isinstance(action_data, dict):
                    action_data = {"raw": str(action_data)}
                
                thoughts_and_actions.append(
                    ThoughtAction(
                        thought=item.get("thought", ""),
                        action=action_data,
                        observation=item.get("observation", "")
                    )
                )
        
        # 로그 위치 정보 생성
        log_session_id = result.get("log_session_id")
        log_path = None
        if log_session_id:
            log_path = f"logs/react_logs/{log_session_id}"
        
        return ChatResponse(
            response=result.get("response", ""),
            thoughts_and_actions=thoughts_and_actions,
            full_response=result.get("full_response", ""),
            error=result.get("error"),
            log_session_id=log_session_id,
            log_path=log_path
        )
    except Exception as e:
        logger.error(f"Error in /chat endpoint: {e}", exc_info=True)
        return ChatResponse(
            response=f"처리 중 오류가 발생했습니다: {str(e)}",
            error=str(e)
        )
