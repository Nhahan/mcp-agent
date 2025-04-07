import logging.config
import logging
import asyncio
import os  # Add os module import
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import sys

from app.api.api import api_router
from app.core.config import get_settings
from app.services.mcp_service import MCPService
from app.services.inference_service import InferenceService

# React 응답 생성용 모델
class ReactInferenceRequest(BaseModel):
    text: str

class ReactResponse(BaseModel):
    generated_text: str

# Configure logging
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    },
    "handlers": {
        "console": {
            "level": "DEBUG", # Set level from env var later
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
    },
    "loggers": {
        "": { # Root logger
            "handlers": ["console"],
            "level": "INFO", # Default level
            "propagate": False,
        },
        "app": { # Logger for the 'app' namespace
            "handlers": ["console"],
            "level": "DEBUG", # Example: Set app logger to DEBUG
            "propagate": False,
        },
        "uvicorn.error": {
            "level": "INFO",
        },
        "uvicorn.access": {
            "handlers": ["console"], # Use console handler for access logs too
            "level": "INFO",
            "propagate": False,
        },
    },
})

logger = logging.getLogger(__name__)

# Application state to hold service instances
app_state = {}

# watchfiles 로거 가져오기 및 레벨 설정
watchfiles_logger = logging.getLogger("watchfiles")
watchfiles_logger.setLevel(logging.WARNING)

# 프로젝트 루트를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application lifespan...")
    settings = get_settings()
    # Dynamically set log level from settings
    log_level = settings.log_level.upper()
    logging.getLogger("app").setLevel(log_level)
    logging.getLogger("").setLevel(log_level) # Adjust root logger level too if needed
    logger.info(f"Log level set to {log_level}")
    
    # Create MCPService instance
    mcp_service = MCPService(settings=settings)
    app_state["mcp_service"] = mcp_service
    
    # Create InferenceService instance, injecting MCPService
    inference_service = InferenceService(settings=settings, mcp_service=mcp_service)
    app_state["inference_service"] = inference_service

    # Start services in the background
    init_task = asyncio.create_task(inference_service.initialize()) # Initialize inference service (includes model download check)
    
    # Optionally, wait for initialization or handle errors
    try:
        # MCP 서버 시작 및 도구 정보 초기화
        logger.info("MCP 서버 시작 중...")
        await mcp_service.start_servers()
        
        # 서버 연결 상태 확인 및 도구 정보 확인
        available_servers = mcp_service.get_available_server_names()
        running_servers = mcp_service.get_running_servers()
        
        logger.info(f"설정에 정의된 MCP 서버: {available_servers}")
        logger.info(f"실제 실행 중인 MCP 서버: {running_servers}")
        
        # 도구 정보 초기화 시도 (더 간결하게 로깅)
        tools_fetched = False
        for name in running_servers:
            # 서버별 연결 시도 및 도구 fetch는 MCPService 내부에서 로깅되므로 여기서는 반복 로깅 최소화
            if mcp_service.is_server_connected(name):
                 tools = mcp_service.get_server_tools(name) # 캐시된 정보 확인
                 if tools: # 캐시된 정보가 있으면 성공으로 간주
                      tools_fetched = True
                 else: # 캐시가 없으면 fetch 시도 (MCPService 내부에서 로그 남김)
                      fetched_tools = await mcp_service._fetch_server_tools(name)
                      if fetched_tools:
                           tools_fetched = True # fetch 성공
                           # 이미 _fetch_server_tools에서 로그 남김
                      # else: # fetch 실패 로그는 _fetch_server_tools 에서 남김
        
        logger.info("서비스 시작 완료")
    except Exception as e:
        logger.error(f"서비스 초기화 중 오류 발생: {e}", exc_info=True)

    yield # Application runs here

    logger.info("Shutting down application lifespan...")
    # Stop MCP servers
    await mcp_service.stop_servers()
    # Cancel any pending tasks if necessary
    if init_task and not init_task.done(): # Check if init_task exists
        init_task.cancel()
    logger.info("Application shutdown complete.")

app = FastAPI(
    title="AI Agent API",
    description="API for the OS-Agnostic AI Agent with MCP integration.",
    version="0.1.0",
    lifespan=lifespan # Use the lifespan context manager
)

# Include the API router at /api/v1 prefix
app.include_router(api_router)

# Add dependency overrides to inject the singleton services
def get_mcp_service_instance():
    return app_state["mcp_service"]

def get_inference_service_instance():
    return app_state["inference_service"]

# Override dependencies used in services/endpoints
app.dependency_overrides[MCPService] = get_mcp_service_instance
app.dependency_overrides[InferenceService] = get_inference_service_instance
# Also override the getter functions if they are directly used as dependencies
from app.services.inference_service import get_inference_service
from app.services.mcp_service import get_mcp_service
app.dependency_overrides[get_inference_service] = get_inference_service_instance
app.dependency_overrides[get_mcp_service] = get_mcp_service_instance

@app.get("/")
async def read_root():
    return {"message": "AI Agent API is running."}

# Keep the local run block if needed for development
if __name__ == "__main__":
    import uvicorn
    import argparse
    
    # 명령행 인수 파싱
    parser = argparse.ArgumentParser(description="AI Agent API 서버")
    parser.add_argument("--host", default="127.0.0.1", help="서버 호스트 (기본값: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="서버 포트 (기본값: 8000)")
    args = parser.parse_args()
    
    logger.warning("Running in debug mode directly with uvicorn. Use Docker for production.")

    # <<<--- 불필요한 setdefault 제거 --- >>>
    # 설정은 config.py와 lifespan에서 처리됨

    # 명시적으로 watchfiles 로거 레벨 설정 (리로드 시 영향 줄 수 있도록)
    watchfiles_logger = logging.getLogger("watchfiles")
    watchfiles_logger.setLevel(logging.WARNING)
    logger.info(f"Explicitly setting watchfiles logger level to WARNING before uvicorn.run")

    # Pass the app instance string for uvicorn's auto-reload feature
    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=True) 