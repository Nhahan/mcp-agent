import logging.config
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pathlib import Path
import sys
from typing import Optional

from app.api.api import api_router
from app.core.config import settings # Import settings directly
from app.services.mcp_service import MCPService
from app.services.inference_service import InferenceService
from app.services.model_service import ModelService

from app.dependencies import (
    get_mcp_service,
    get_inference_service,
    get_model_service,
    set_mcp_service_instance,
    set_inference_service_instance,
    set_model_service_instance
)

# Configure logging based on settings
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
            "level": settings.log_level.upper(), # Use settings log level
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
    },
    "loggers": {
        "": { # Root logger
            "handlers": ["console"],
            "level": settings.log_level.upper(), # Use settings log level
            "propagate": False,
        },
        "app": { # Logger for the 'app' namespace
            "handlers": ["console"],
            "level": settings.log_level.upper(), # Use settings log level
            "propagate": False,
        },
        "uvicorn.error": {
            "handlers": ["console"],
            "level": "INFO", # Keep uvicorn loggers at INFO
            "propagate": False,
        },
        "uvicorn.access": {
            "handlers": ["console"],
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

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager. 
    Handles initialization and cleanup of core services:
    - ModelService for LLM operations
    - MCPService for Model Context Protocol servers
    - InferenceService combining both for the ReAct pattern
    """
    logger.info("Starting application lifespan...")

    # Create log directory
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Logging initialized. Log level: {settings.log_level.upper()}, Log directory: {log_dir}")

    # Initialize core service instances
    model_service: Optional[ModelService] = None
    mcp_service: Optional[MCPService] = None
    inference_service: Optional[InferenceService] = None
    
    try:
        # 1. ModelService - handles LLM interactions
        logger.info(f"Initializing ModelService with model path: {settings.model_path}")
        # Model params come directly from settings now
        model_service = ModelService() # No arguments needed; will use settings
        set_model_service_instance(model_service) # Store instance for DI
        if model_service.model_loaded:
            logger.info("ModelService initialized and model loaded successfully.")
        else:
            logger.warning("ModelService initialized BUT model loading failed or was deferred. Some endpoints may not work.")

        # 2. MCPService - handles MCP server management
        logger.info(f"Initializing MCPService with config path: {settings.mcp_config_path}")
        mcp_service = MCPService() # No settings parameter needed now
        await mcp_service.start_servers()
        set_mcp_service_instance(mcp_service) # Store instance for DI
        logger.info(f"MCPService started with {len(mcp_service.list_servers() or [])} servers.")

        # 3. InferenceService - combines ModelService and MCPService for ReAct pattern
        logger.info("Initializing InferenceService with ModelService and MCPService...")
        if model_service and mcp_service: # Ensure dependencies are created
            inference_service = InferenceService(
                mcp_manager=mcp_service,
                model_service=model_service
            )
            # Log dir is now handled by settings
            set_inference_service_instance(inference_service) # Store instance for DI
            logger.info("InferenceService initialized successfully.")
        else:
            logger.error("Cannot initialize InferenceService due to missing ModelService or MCPService.")
            raise RuntimeError("Failed to initialize core services (Model/MCP) needed by InferenceService.")

        # Set dependency overrides to use the initialized instances
        app.dependency_overrides[get_model_service] = lambda: model_service
        app.dependency_overrides[get_mcp_service] = lambda: mcp_service
        app.dependency_overrides[get_inference_service] = lambda: inference_service
        logger.info("FastAPI dependency overrides configured.")

        logger.info(f"Application startup completed successfully. API version: {settings.api_version}")

    except Exception as e:
        logger.critical(f"Fatal error during application startup: {e}", exc_info=True)
        # Attempt cleanup even on startup failure
        if mcp_service:
            try: 
                logger.info("Attempting MCPService shutdown due to startup failure...")
                await mcp_service.stop_servers()
            except Exception as cleanup_e:
                 logger.error(f"Error during MCPService cleanup after startup failure: {cleanup_e}", exc_info=True)
        if model_service:
            try:
                logger.info("Attempting ModelService shutdown...")
                await model_service.shutdown()
            except Exception as cleanup_e:
                logger.error(f"Error during ModelService cleanup: {cleanup_e}", exc_info=True)
        # Clean up dependency instances even on failure
        set_model_service_instance(None)
        set_mcp_service_instance(None)
        set_inference_service_instance(None)
        raise # Re-raise to stop the application

    # Yield control back to FastAPI while application runs
    yield 

    # --- Shutdown Logic --- 
    logger.info("Application shutdown initiated...")
    
    # Stop MCP servers
    if mcp_service:
        logger.info("Shutting down MCP Service...")
        try:
            await mcp_service.stop_servers()
            logger.info("MCP Service shutdown completed.")
        except Exception as e:
            logger.error(f"Error during MCPService shutdown: {e}", exc_info=True)
    
    # Release model resources
    if model_service:
        logger.info("Shutting down Model Service...")
        try:
            await model_service.shutdown() 
            logger.info("Model Service shutdown completed.")
        except Exception as e:
            logger.error(f"Error during ModelService shutdown: {e}", exc_info=True)
    
    # Clear dependency instances
    set_model_service_instance(None)
    set_mcp_service_instance(None)
    set_inference_service_instance(None)
    logger.info("Dependency instances cleared.")
    logger.info("Application shutdown complete.")

# Create FastAPI application
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan 
)

# Include API router
app.include_router(api_router)

@app.get("/")
async def read_root():
    """Root endpoint that confirms the API is running."""
    return {
        "status": "active",
        "message": f"{settings.api_title} v{settings.api_version} is running",
        "docs_url": "/docs"
    }

# --- Main execution block (for direct execution with Python) --- 
if __name__ == "__main__":
    import uvicorn
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description=settings.api_description)
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    args = parser.parse_args()
    
    logger.warning("Running in debug mode directly with uvicorn. Use Docker for production.")

    # Start the API server
    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=False) 