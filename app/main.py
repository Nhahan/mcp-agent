import logging.config
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pathlib import Path
import sys

from app.api.api import api_router
from app.core.config import get_settings, Settings
from app.services.mcp_service import MCPService
from app.services.inference_service import InferenceService

# dependencies.py 에서 필요한 것들 임포트
from app.dependencies import (
    get_mcp_service,
    get_inference_service,
    set_mcp_service_instance,
    set_inference_service_instance,
    _mcp_service_instance, # 종료 시 None 확인용
    _inference_service_instance # 종료 시 None 확인용
)

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
            "level": "DEBUG", # Default to DEBUG for console
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
    },
    "loggers": {
        "": { # Root logger
            "handlers": ["console"],
            "level": "DEBUG", # Set root level to DEBUG
            "propagate": False,
        },
        "app": { # Logger for the 'app' namespace
            "handlers": ["console"],
            "level": "DEBUG", # Ensure app logger is DEBUG
            "propagate": False,
        },
        "uvicorn.error": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn.access": {
            "handlers": ["console"], # Use console handler for access logs too
            "level": "INFO",
            "propagate": False,
        },
        "watchfiles": { # Quieten watchfiles
             "handlers": ["console"],
             "level": "WARNING",
             "propagate": False,
        }
    },
})

logger = logging.getLogger(__name__)

# Set watchfiles logger level
watchfiles_logger = logging.getLogger("watchfiles")
watchfiles_logger.setLevel(logging.WARNING)

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application lifespan...")
    settings = get_settings()

    # --- 로깅 설정 (변경 없음) --- 
    log_level = settings.log_level.upper()
    logging.getLogger("app").setLevel(log_level)
    logging.getLogger("").setLevel(log_level)
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Logging initialized. Log level: {log_level}, Log directory: {log_dir}")

    # --- 서비스 인스턴스 생성 및 dependencies.py 에 설정 --- 
    # 1. MCPService 생성 및 시작
    mcp_service = MCPService(settings=settings)
    await mcp_service.start_servers()
    set_mcp_service_instance(mcp_service) # dependencies.py 에 설정

    # 2. InferenceService 생성 (MCPService 주입)
    model_path = settings.model_path
    model_params = {
        "n_ctx": settings.n_ctx, 
        "n_gpu_layers": settings.gpu_layers,
    }
    inference_service = InferenceService(
        mcp_manager=mcp_service, # 로컬 변수 사용
        model_path=model_path,
        model_params=model_params
    )
    inference_service.set_log_directory(log_dir)
    set_inference_service_instance(inference_service) # dependencies.py 에 설정
    
    # --- 의존성 주입 오버라이드 설정 --- 
    # dependencies.py 에서 가져온 getter 함수 사용
    app.dependency_overrides[get_mcp_service] = get_mcp_service
    app.dependency_overrides[get_inference_service] = get_inference_service

    try:
        # --- 시작 작업 ---
        # 모델 파일 존재 여부 확인
        model_path = Path(settings.model_path)
        if not model_path.exists():
            logger.critical(f"모델 파일을 찾을 수 없습니다: {model_path}")
            logger.critical(f"필요한 모델 파일을 ./models 디렉토리에 수동으로 다운로드하세요")
            raise FileNotFoundError(f"필요한 모델 파일을 찾을 수 없습니다: {model_path}")
        
        logger.info(f"모델 파일을 찾았습니다: {model_path}")
        logger.info("LLM 모델 초기화 중...")
        await inference_service.initialize_model() # 로컬 변수 사용
        # --- 시작 작업 끝 ---
 
        logger.info("애플리케이션 시작 작업이 성공적으로 완료되었습니다.")
    except Exception as e:
        logger.critical(f"Fatal error during application startup: {e}", exc_info=True)
        # 예외를 다시 발생시켜 앱이 종료되도록 함
        raise

    yield # Application runs here

    logger.info("Shutting down application lifespan...")
    # --- 종료 작업 --- 
    # dependencies.py 에서 가져온 인스턴스 변수 사용 (_inference_service_instance)
    inf_service_to_shutdown = _inference_service_instance
    mcp_service_to_shutdown = _mcp_service_instance
    
    if inf_service_to_shutdown:
        logger.info("Shutting down Inference Service (releasing model)...")
        await inf_service_to_shutdown.shutdown_model()
    if mcp_service_to_shutdown:
        logger.info("Shutting down MCP Service...")
        await mcp_service_to_shutdown.stop_servers()
    # --- 종료 작업 끝 ---
    logger.info("Application shutdown complete.")

app = FastAPI(
    title="AI Agent API",
    description="API for the OS-Agnostic AI Agent with MCP integration.",
    version="0.1.0",
    lifespan=lifespan 
)

# Include the API router
app.include_router(api_router)

# --- 의존성 주입 함수 정의 제거 --- 
# def get_mcp_service() -> MCPService:
#      ...
# 
# def get_inference_service() -> InferenceService:
#      ...

@app.get("/")
async def read_root():
    return {"message": "AI Agent API is running."}

# --- 메인 실행 블록 (변경 없음) --- 
if __name__ == "__main__":
    # ... (uvicorn 실행 코드)
    import uvicorn
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AI Agent API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    args = parser.parse_args()
    
    logger.warning("Running in debug mode directly with uvicorn. Use Docker for production.")

    # Explicitly set watchfiles logger level (to affect reloads)
    watchfiles_logger = logging.getLogger("watchfiles")
    watchfiles_logger.setLevel(logging.WARNING)
    logger.info(f"Explicitly setting watchfiles logger level to WARNING before uvicorn.run")

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=False) 