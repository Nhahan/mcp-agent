import logging.config
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import sys
from typing import Optional

from app.api.api import api_router
from app.core.config import settings
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
    log_dir = settings.calculated_log_dir_path
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Logging initialized. Log level: {settings.log_level.upper()}, Log directory: {log_dir}")

    # Initialize core service instances
    model_service_local: Optional[ModelService] = None
    mcp_service_local: Optional[MCPService] = None
    inference_service_local: Optional[InferenceService] = None
    
    try:
        # 1. ModelService - handles LLM interactions
        logger.info(f"Initializing ModelService with model path: {settings.calculated_model_path}")
        model_service_local = ModelService() # No arguments needed; will use settings
        set_model_service_instance(model_service_local) # Store instance for DI
        if model_service_local.model_loaded:
            logger.info("ModelService initialized and model loaded successfully.")
        else:
            logger.warning("ModelService initialized BUT model loading failed or was deferred. Some endpoints may not work.")
        mcp_service_local = MCPService() # No settings parameter needed now
        await mcp_service_local.start_servers()
        set_mcp_service_instance(mcp_service_local) # Store instance for DI
        logger.info(f"MCPService started with {len(mcp_service_local.list_servers() or [])} servers.")

        # 3. InferenceService - combines ModelService and MCPService for ReAct pattern
        logger.info("Initializing InferenceService with ModelService and MCPService...")
        if model_service_local and mcp_service_local: # Ensure dependencies are created
            inference_service_local = InferenceService(
                mcp_manager=mcp_service_local,
                model_service=model_service_local
            )
            # Log dir is now handled by settings
            set_inference_service_instance(inference_service_local) # Store instance for DI
            logger.info("InferenceService initialized successfully.")
        else:
            logger.error("Cannot initialize InferenceService due to missing ModelService or MCPService.")
            raise RuntimeError("Failed to initialize core services (Model/MCP) needed by InferenceService.")

        # Set dependency overrides using the GETTER functions directly.
        # FastAPI knows how to call these functions when the dependency is needed.
        app.dependency_overrides[get_model_service] = get_model_service
        app.dependency_overrides[get_mcp_service] = get_mcp_service
        app.dependency_overrides[get_inference_service] = get_inference_service
        logger.info("FastAPI dependency overrides configured.")

        logger.info(f"Application startup completed. API: {settings.api_v1_prefix}")

    except Exception as e:
        logger.critical(f"Fatal error during application startup: {e}", exc_info=True)
        # Attempt cleanup even on startup failure
        if mcp_service_local:
            try: 
                logger.info("Attempting MCPService shutdown due to startup failure...")
                await mcp_service_local.stop_servers()
            except Exception as cleanup_e:
                 logger.error(f"Error during MCPService cleanup after startup failure: {cleanup_e}", exc_info=True)
        if model_service_local:
            try:
                logger.info("Attempting ModelService shutdown...")
                await model_service_local.shutdown()
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
    
    # Retrieve services from app state
    mcp_service_shutdown: Optional[MCPService] = app.state.mcp_service if hasattr(app.state, 'mcp_service') else None
    model_service_shutdown: Optional[ModelService] = app.state.model_service if hasattr(app.state, 'model_service') else None

    # Stop MCP servers
    if mcp_service_shutdown:
        logger.info("Shutting down MCP Service...")
        try:
            await mcp_service_shutdown.stop_servers()
            logger.info("MCP Service shutdown completed.")
        except Exception as e:
            logger.error(f"Error during MCPService shutdown: {e}", exc_info=True)
    else:
        logger.warning("MCPService instance not found in app state during shutdown.")
    
    # Release model resources
    if model_service_shutdown:
        logger.info("Shutting down Model Service...")
        try:
            await model_service_shutdown.shutdown() 
            logger.info("Model Service shutdown completed.")
        except Exception as e:
            logger.error(f"Error during ModelService shutdown: {e}", exc_info=True)
    else:
        logger.warning("ModelService instance not found in app state during shutdown.")

    # Clear dependency instances (already handled by dependency.py functions potentially)
    # set_model_service_instance(None) 
    # set_mcp_service_instance(None)
    # set_inference_service_instance(None)
    # logger.info("Dependency instances cleared.") # Logging this might be redundant
    logger.info("Application shutdown complete.")

# Create FastAPI application
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan,
    docs_url=f"{settings.api_v1_prefix}/docs",
    redoc_url=f"{settings.api_v1_prefix}/redoc",
    openapi_url=f"{settings.api_v1_prefix}/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Include API router
app.include_router(api_router, prefix=settings.api_v1_prefix)

# Define the path to the static files directory (inside the app dir in Docker)
# For local dev, this dir might not exist unless frontend is built
static_files_dir = Path(__file__).parent.parent / "static" 
assets_dir = static_files_dir / "assets"
index_html_path = static_files_dir / "index.html"

# Only mount static files if the directories/files exist (for production/Docker)
if assets_dir.is_dir():
    logger.info(f"Mounting static assets from: {assets_dir}")
    app.mount("/assets", StaticFiles(directory=assets_dir, check_dir=True), name="assets") # check_dir is True by default
else:
    logger.warning(f"Assets directory not found ({assets_dir}), skipping /assets mount. This is expected in development.")

if index_html_path.is_file():
    logger.info(f"Mounting static root and serving index.html from: {static_files_dir}")
    # Serve files like favicon.ico from the root of static_files_dir
    # and serve index.html for the root path `/`.
    # Using check_dir=False because we only care if index.html exists here.
    app.mount("/static-root", StaticFiles(directory=static_files_dir, check_dir=False), name="static_root_files") 

    # Catch-all to serve index.html for client-side routing
    @app.get("/{full_path:path}")
    async def serve_frontend_catch_all(full_path: str):
        # Exclude API paths explicitly to avoid conflicts 
        if full_path.startswith(settings.api_v1_prefix.strip('/')) or \
           full_path.startswith('assets/') or \
           full_path.startswith('static-root/'): # Check against mount paths
             # Let FastAPI handle API/static routes or return 404 if they don't match
             # We raise HTTPException here which will be caught by FastAPI's default handling
             # for non-matching routes, resulting in a 404.
             raise HTTPException(status_code=404, detail="Resource not found")
        
        logger.debug(f"Serving index.html for path: {full_path}")
        return FileResponse(index_html_path)

else:
    logger.warning(f"index.html not found ({index_html_path}), skipping static file serving. Frontend must be served separately (e.g., npm run dev). Falling back to basic root endpoint.")
    # Fallback root endpoint if index.html is not found (useful for dev)
    @app.get("/")
    async def read_root_fallback():
        return {
            "status": "active",
            "message": f"{settings.api_title} v{settings.api_version} is running (API only - frontend not served)",
            "api_docs_url": f"{settings.api_v1_prefix}/docs"
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