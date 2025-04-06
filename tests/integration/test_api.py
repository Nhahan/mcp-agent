import pytest
import pytest_asyncio
import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Import the FastAPI app instance
# Ensure PYTHONPATH allows importing 'app'
from app.main import app

@pytest.fixture(scope="module")
def test_client() -> TestClient:
    """Create a TestClient instance for the FastAPI app."""
    # Using TestClient directly bypasses lifespan events unless wrapped in 'with'
    return TestClient(app)

# Use pytest_asyncio.fixture for async fixtures
@pytest_asyncio.fixture(scope="module")
async def async_test_client() -> httpx.AsyncClient:
    """Create an httpx AsyncClient for testing the app using ASGITransport."""
    # Use ASGITransport to wrap the FastAPI app for httpx
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client # Yield the client instance

# Test the root status endpoint (sync)
def test_read_main_v1_status_sync(test_client: TestClient):
    """Test the /v1/ status endpoint using TestClient."""
    response = test_client.get("/v1/")
    assert response.status_code == 200
    assert response.json() == {"message": "AI Agent is running."}

# Test the root status endpoint (async)
@pytest.mark.asyncio
async def test_read_main_v1_status_async(async_test_client: httpx.AsyncClient): # Fixture provides the client directly
    """Test the /v1/ status endpoint using httpx.AsyncClient."""
    # No need to await the fixture itself, just use the yielded client
    response = await async_test_client.get("/v1/")
    assert response.status_code == 200
    assert response.json() == {"message": "AI Agent is running."}

# Test the root redirect (sync)
def test_root_redirect_sync(test_client: TestClient):
    """Test the redirect from / to /v1/."""
    # follow_redirects=False to check the redirect status code
    response = test_client.get("/", follow_redirects=False)
    assert response.status_code == 307 # Temporary Redirect (default by FastAPI)
    assert response.headers["location"] == "/v1/"

# Test the root redirect (async)
@pytest.mark.asyncio
async def test_root_redirect_async(async_test_client: httpx.AsyncClient): # Fixture provides the client directly
    """Test the redirect from / to /v1/ using httpx."""
    # follow_redirects=False to check the redirect status code
    response = await async_test_client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/v1/"

# Example placeholder for inference endpoint test (requires more setup/mocking)
# @pytest.mark.asyncio
# async def test_inference_endpoint(async_test_client: httpx.AsyncClient):
#     # This test would require mocking the model session or providing
#     # a dummy model for testing purposes, as the real model download/
#     # loading is likely too slow and resource-intensive for regular tests.
#     # Also requires defining expected input/output based on the placeholder.
#     pass

# Example placeholder for MCP status endpoint test
# @pytest.mark.asyncio
# async def test_mcp_status_endpoint(async_test_client: httpx.AsyncClient):
#     # This test would require mocking the mcp_service functions
#     # (e.g., get_mcp_process) or potentially running a dummy MCP process.
#     pass 