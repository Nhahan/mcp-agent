from fastapi import APIRouter

router = APIRouter()

@router.get("/", tags=["Status"], summary="Root endpoint for basic status check")
async def read_root():
    """Provides a simple message indicating the service is running."""
    return {"message": "AI Agent is running."}

# Add other general status endpoints if needed 