import pytest
import os
import shutil
import time
from pathlib import Path
import subprocess
import logging
import json

from fastapi.testclient import TestClient
# from app.mcp_client.client import MCPError # No longer needed directly in tests

pytestmark = pytest.mark.e2e
logger = logging.getLogger(__name__)

# --- Test Configuration ---
TEST_MODEL_DIR = Path("test_model_cache")
TEST_MODEL_FILENAME = "model_q4f16.onnx"
TEST_MODEL_PATH = TEST_MODEL_DIR / TEST_MODEL_FILENAME
DEFAULT_MODEL_URL = "https://huggingface.co/onnx-community/gemma-3-1b-it-ONNX/resolve/main/onnx/model_q4f16.onnx?download=true"
MCP_CONFIG_PATH = Path("mcp.json")

@pytest.fixture(scope="module", autouse=True)
def setup_test_environment():
    logger.info("Setting up E2E test environment...")
    logger.info("Ensuring MCP config exists...")
    if not MCP_CONFIG_PATH.exists():
        # Create a dummy mcp.json if it doesn't exist, required for startup
        # IMPORTANT: Tests requiring specific MCP servers need to handle their setup/availability
        logger.warning(f"{MCP_CONFIG_PATH} not found. Creating a dummy one.")
        with open(MCP_CONFIG_PATH, "w") as f:
            json.dump({"mcpServers": {}}, f)
    # Clear model cache before tests
    # logger.info("Cleaning up model cache...")
    # if TEST_MODEL_DIR.exists(): 
    #     shutil.rmtree(TEST_MODEL_DIR)
    TEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    yield
    logger.info("Tearing down E2E test environment...")
    # Optional: Clean up dummy mcp.json if created
    # if MCP_CONFIG_PATH.exists() and Path(MCP_CONFIG_PATH).read_text() == '{"mcpServers": {}}':
    #     MCP_CONFIG_PATH.unlink()

@pytest.fixture(scope="module")
def e2e_test_client(setup_test_environment):
    original_env = os.environ.copy()
    env_vars_to_set = {
        "MODEL_URL": os.getenv("TEST_MODEL_URL", DEFAULT_MODEL_URL),
        "MODEL_PATH": str(TEST_MODEL_PATH.resolve()),
        "MCP_CONFIG_PATH": str(MCP_CONFIG_PATH.resolve()),
        "LOG_LEVEL": "DEBUG"
    }
    logger.debug(f"Setting E2E test env vars: {env_vars_to_set}")
    os.environ.update(env_vars_to_set)
    # Ensure settings are reloaded with new env vars
    from app.core.config import get_settings
    get_settings.cache_clear()
    from app.main import app
    client = None
    try:
        logger.info("Starting TestClient with lifespan...")
        with TestClient(app) as client:
            logger.info("Waiting for app startup (model download & MCP init)... ")
            # Increase wait time significantly as model download + MCP startup can be long
            wait_time = 300 # 5 minutes, adjust as needed
            start_wait = time.time()
            # Check model file and potentially basic MCP server status (if needed and possible)
            # For simplicity, just wait for model file as a proxy for readiness
            while not TEST_MODEL_PATH.is_file() and time.time() - start_wait < wait_time:
                logger.info(f"Waiting for model... ({int(time.time() - start_wait)}s)")
                time.sleep(15)
            if not TEST_MODEL_PATH.is_file():
                pytest.fail("Model download timeout during test setup.")
            logger.info("Model file found. Assuming agent is ready.")
            # Add a small buffer for MCP servers to fully initialize
            time.sleep(10)
            yield client
    finally:
        logger.info("Restoring environment...")
        os.environ.clear(); os.environ.update(original_env)
        get_settings.cache_clear()
        logger.info("TestClient finished.")

@pytest.fixture(scope="module")
def _check_model_file_present(e2e_test_client):
    logger.info(f"Checking model file: {TEST_MODEL_PATH}")
    if not TEST_MODEL_PATH.is_file(): pytest.fail("Model not found.")
    logger.info("Model found.")
    return True

def test_e2e_model_downloaded(_check_model_file_present):
    assert _check_model_file_present is True

def test_e2e_normal_inference(e2e_test_client: TestClient, _check_model_file_present):
    """Test standard LLM inference without triggering MCP tools."""
    assert _check_model_file_present is True
    test_prompt = "Explain the importance of AI agents in one sentence:"
    payload = {"text": test_prompt}
    response = e2e_test_client.post("/v1/inference", json=payload)
    assert response.status_code == 200
    response_data = response.json()
    generated_text = response_data["generated_text"]
    assert "Terminal Output:" not in generated_text
    assert len(generated_text) > 5
    logger.info(f"Normal inference check passed.")

@pytest.mark.skipif(not shutil.which("npx"), reason="npx (Node.js) is required for iterm-mcp")
def test_e2e_inference_with_iterm_mcp(e2e_test_client: TestClient, _check_model_file_present):
    """Test inference that should trigger iterm-mcp execution."""
    assert _check_model_file_present is True
    # Ensure iterm-mcp is configured in mcp.json for this test
    try:
        with open(MCP_CONFIG_PATH, "r") as f:
            mcp_config = json.load(f)
        if "iterm-mcp" not in mcp_config.get("mcpServers", {}):
            pytest.skip("iterm-mcp not found in mcp.json configuration.")
    except FileNotFoundError:
        pytest.skip(f"{MCP_CONFIG_PATH} not found.")
    except json.JSONDecodeError:
        pytest.fail(f"Error decoding {MCP_CONFIG_PATH}.")

    unique_echo_string = f"E2E_MCP_TEST_{int(time.time())}"
    test_prompt = f"터미널에서 echo {unique_echo_string} 실행해줘"
    payload = {"text": test_prompt}
    response = e2e_test_client.post("/v1/inference", json=payload)
    assert response.status_code == 200
    response_data = response.json()
    generated_text = response_data["generated_text"]

    # Verify that the response contains the expected markers of terminal output
    assert generated_text.startswith("Terminal Output:")
    assert unique_echo_string in generated_text
    logger.info("Inference with iterm-mcp check passed.")

# Remove the old MCP direct call tests
# def test_e2e_iterm_mcp_status(...):
# def test_e2e_iterm_mcp_tools(...): 