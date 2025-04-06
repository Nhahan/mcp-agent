from fastapi import APIRouter

from app.api.v1.endpoints import inference # Keep inference endpoint
# from app.api.v1.endpoints import mcp # Remove MCP endpoint import

api_router = APIRouter()

# Include routers from endpoint modules
# api_router.include_router(status.router) # Remove this line related to old status endpoint
api_router.include_router(inference.router, prefix="/inference", tags=["inference"])
# api_router.include_router(mcp.router, prefix="/mcp", tags=["mcp"]) # MCP router is already removed

# You can add prefix='/v1' here or in main.py when including this router 