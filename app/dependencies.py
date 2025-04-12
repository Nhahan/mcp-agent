from typing import Optional
import logging

# Import settings and service classes
from app.core.config import settings
from app.services.mcp_service import MCPService
from app.services.inference_service import InferenceService
from app.services.model_service import ModelService

logger = logging.getLogger(__name__)

# --- Singleton instance storage variables ---
_mcp_service_instance: Optional[MCPService] = None
_inference_service_instance: Optional[InferenceService] = None
_model_service_instance: Optional[ModelService] = None

def set_mcp_service_instance(instance: Optional[MCPService]) -> None:
    """Sets the global MCPService instance."""
    global _mcp_service_instance
    _mcp_service_instance = instance
    logger.debug(f"MCP service instance {'set' if instance else 'cleared'}")

def set_inference_service_instance(instance: Optional[InferenceService]) -> None:
    """Sets the global InferenceService instance."""
    global _inference_service_instance
    _inference_service_instance = instance
    logger.debug(f"Inference service instance {'set' if instance else 'cleared'}")

def set_model_service_instance(instance: Optional[ModelService]) -> None:
    """Sets the global ModelService instance."""
    global _model_service_instance
    _model_service_instance = instance
    logger.debug(f"Model service instance {'set' if instance else 'cleared'}")

# --- Dependency injection functions (Getters) ---
def get_mcp_service() -> MCPService:
    """
    Returns the singleton MCPService instance.
    This is used by FastAPI for dependency injection.
    """
    if _mcp_service_instance is None:
        # This error means the lifespan context hasn't properly initialized services
        logger.error("MCPService requested but not initialized. Application lifespan might not have run properly.")
        raise RuntimeError("MCPService has not been initialized. Ensure the application lifespan context is correctly set up.")
    return _mcp_service_instance

def get_inference_service() -> InferenceService:
    """
    Returns the singleton InferenceService instance.
    This is used by FastAPI for dependency injection.
    """
    if _inference_service_instance is None:
        logger.error("InferenceService requested but not initialized. Application lifespan might not have run properly.")
        raise RuntimeError("InferenceService has not been initialized. Ensure the application lifespan context is correctly set up.")
    return _inference_service_instance

def get_model_service() -> ModelService:
    """
    Returns the singleton ModelService instance.
    This is used by FastAPI for dependency injection.
    """
    if _model_service_instance is None:
        logger.error("ModelService requested but not initialized. Application lifespan might not have run properly.")
        raise RuntimeError("ModelService has not been initialized. Ensure the application lifespan context is correctly set up.")
    return _model_service_instance 