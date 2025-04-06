from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel
import logging

from app.services.inference_service import InferenceService, get_inference_service

logger = logging.getLogger(__name__)

router = APIRouter()

class InferenceRequest(BaseModel):
    text: str

# Simplify response model as the service now returns a simple string
class SimpleTextResponse(BaseModel):
    generated_text: str

# Update response_model in the endpoint decorator
@router.post("/", response_model=SimpleTextResponse)
async def run_inference_endpoint(
    request: InferenceRequest,
    inference_service: InferenceService = Depends(get_inference_service)
):
    """
    Run inference using the loaded ONNX model via InferenceService.
    The service might return direct LLM output or formatted MCP tool output.
    """
    logger.info(f"Received inference request for text: {request.text[:50]}...")
    try:
        # Call the generate method which returns a string
        generated_text_str = await inference_service.generate(request.text)
        logger.info("Inference/Agent logic successful.")
        # Return the string in the simplified response model format
        return SimpleTextResponse(generated_text=generated_text_str)
    except RuntimeError as e:
        logger.error(f"Inference runtime error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        logger.error(f"Inference value error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during inference: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during inference.") 