from fastapi import APIRouter, Depends
from pydantic import BaseModel
import logging
import os
from pathlib import Path

from app.services.inference_service import InferenceService, get_inference_service
from app.services.mcp_service import MCPService, get_mcp_service

logger = logging.getLogger(__name__)

router = APIRouter()

class StatusResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    model_exists: bool
    mcp_servers: list
    version: str = "0.1.0"

@router.get("/", response_model=StatusResponse)
async def get_status(
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
    
    logger.info(f"Status check: model_loaded={model_loaded}, model_exists={model_exists}")
    
    return StatusResponse(
        status="ok",
        model_loaded=model_loaded,
        model_path=str(model_path),
        model_exists=model_exists,
        mcp_servers=mcp_status
    ) 