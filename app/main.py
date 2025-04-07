import logging.config
import logging
import asyncio
import os  # Add os module import
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.api.v1.api import api_router
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
    mcp_task = asyncio.create_task(mcp_service.start_servers())
    init_task = asyncio.create_task(inference_service.initialize()) # Initialize inference service (includes model download check)
    
    # Optionally, wait for initialization or handle errors
    try:
        # Wait for essential initializations if needed, or just let them run
        # await init_task # Uncomment if you need model download before serving requests
        logger.info("Services started in background.")
    except Exception as e:
        logger.error(f"Error during service initialization: {e}", exc_info=True)
        # Decide how to handle startup errors (e.g., exit, retry)

    yield # Application runs here

    logger.info("Shutting down application lifespan...")
    # Stop MCP servers
    await mcp_service.stop_servers()
    # Cancel any pending tasks if necessary
    if not mcp_task.done():
        mcp_task.cancel()
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
app.include_router(api_router, prefix="/api/v1")

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
    return {"message": "AI Agent API is running. Use /api/v1/chat for the chat endpoint."}

# Keep the local run block if needed for development
if __name__ == "__main__":
    import uvicorn
    import argparse
    
    # 명령행 인수 파싱
    parser = argparse.ArgumentParser(description="AI Agent API 서버")
    parser.add_argument("--host", default="127.0.0.1", help="서버 호스트 (기본값: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="서버 포트 (기본값: 8000)")
    args = parser.parse_args()
    
    # Logging will be configured by lifespan when uvicorn starts the app
    logger.warning("Running in debug mode directly with uvicorn. Use Docker for production.")

    # Define default environment variables ONLY if they are not already set
    os.environ.setdefault("MODEL_URL", "https://huggingface.co/google/gemma-3-1b-it-qat-q4_0-gguf/resolve/main/gemma-3-1b-it-q4_0.gguf?download=true")
    os.environ.setdefault("MODEL_PATH", "model/gemma-3-1b-it-q4_0.gguf")
    os.environ.setdefault("MCP_CONFIG_PATH", "mcp.json")
    os.environ.setdefault("LOG_LEVEL", "DEBUG") # Example for local dev

    # No need to reload settings here, lifespan will handle it.

    # Pass the app instance string for uvicorn's auto-reload feature
    # Uvicorn will call the lifespan manager upon startup
    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=True)
    # log_level is now controlled by the lifespan based on LOG_LEVEL env var 