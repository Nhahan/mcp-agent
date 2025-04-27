# tests/core/test_llm_loader.py
import os
import pytest
from unittest.mock import patch, MagicMock
from dotenv import set_key, find_dotenv

# Import the module/function to test
from core.llm_loader import load_llm, llm_instance as loader_llm_instance # Import global instance too
from langchain_community.llms import LlamaCpp

# Find .env file to modify during tests
dotenv_path = find_dotenv()
if not dotenv_path:
    # If .env doesn't exist, create one for testing purposes in the current dir
    dotenv_path = ".env_test_llm_loader"
    with open(dotenv_path, "w") as f:
        f.write("# Test env file\n")

original_model_path = os.getenv("MODEL_PATH")

@pytest.fixture(autouse=True)
def reset_llm_instance():
    """ Ensure llm_instance is reset before each test """
    global loader_llm_instance
    loader_llm_instance = None
    # Reset MODEL_PATH in environment and .env file if it was changed
    if original_model_path:
        os.environ["MODEL_PATH"] = original_model_path
        set_key(dotenv_path, "MODEL_PATH", original_model_path)
    elif "MODEL_PATH" in os.environ:
        del os.environ["MODEL_PATH"]
        # Note: Removing key from .env file reliably is tricky without parsing libraries
        # For simplicity, we just ensure it's not set in os.environ if it wasn't originally
    yield
    # Cleanup after test if needed
    loader_llm_instance = None
    if os.path.exists(".env_test_llm_loader"):
        os.remove(".env_test_llm_loader")


def test_load_llm_missing_env_variable():
    """ Test ValueError is raised if MODEL_PATH is not set """
    if "MODEL_PATH" in os.environ:
        del os.environ["MODEL_PATH"] # Ensure it's not set
    # Also try removing from the .env file for robustness
    set_key(dotenv_path, "MODEL_PATH", "") # Set to empty string

    with pytest.raises(ValueError, match="MODEL_PATH environment variable not set."):
        load_llm()

def test_load_llm_invalid_path():
    """ Test ValueError is raised if MODEL_PATH points to a non-existent file """
    invalid_path = "./non_existent_model.gguf"
    set_key(dotenv_path, "MODEL_PATH", invalid_path)
    os.environ["MODEL_PATH"] = invalid_path # Ensure env var is updated

    with pytest.raises(ValueError, match=f"Model file not found at path: {invalid_path}"):
        load_llm()

# --- Tests requiring a valid model path ---
# Set SKIP_REAL_MODEL_TESTS=1 in environment to skip these
SKIP_REAL_MODEL_TESTS = os.getenv("SKIP_REAL_MODEL_TESTS", "0") == "1"
VALID_MODEL_PATH_EXISTS = original_model_path and os.path.exists(original_model_path)

@pytest.mark.skipif(SKIP_REAL_MODEL_TESTS or not VALID_MODEL_PATH_EXISTS, reason="Requires valid MODEL_PATH and model file")
def test_load_llm_success():
    """ Test successful loading returns a LlamaCpp instance """
    # Ensure the original valid path is used
    set_key(dotenv_path, "MODEL_PATH", original_model_path)
    os.environ["MODEL_PATH"] = original_model_path

    llm = load_llm()
    assert isinstance(llm, LlamaCpp)
    assert llm.model_path == original_model_path

@pytest.mark.skipif(SKIP_REAL_MODEL_TESTS or not VALID_MODEL_PATH_EXISTS, reason="Requires valid MODEL_PATH and model file")
def test_load_llm_singleton():
    """ Test that loading multiple times returns the same instance """
    set_key(dotenv_path, "MODEL_PATH", original_model_path)
    os.environ["MODEL_PATH"] = original_model_path

    llm1 = load_llm()
    llm2 = load_llm()
    assert llm1 is llm2
    assert isinstance(llm1, LlamaCpp)

# Optional: Test actual inference (can be slow and requires resources)
@pytest.mark.skipif(True, reason="Actual inference test disabled by default")
# @pytest.mark.skipif(SKIP_REAL_MODEL_TESTS or not VALID_MODEL_PATH_EXISTS, reason="Requires valid MODEL_PATH and model file")
def test_load_llm_inference():
    """ Test basic model inference """
    set_key(dotenv_path, "MODEL_PATH", original_model_path)
    os.environ["MODEL_PATH"] = original_model_path
    llm = load_llm()
    try:
        # Use a simple, short prompt
        response = llm.invoke("What is 1 + 1?")
        print(f"\nInference Response: {response}") # Print response for debugging
        assert isinstance(response, str)
        assert len(response) > 0
        # Add more specific checks if needed, e.g., check if '2' is in the response
        # assert "2" in response or "two" in response.lower()
    except Exception as e:
        pytest.fail(f"Model invocation failed: {e}")

# Test mocking LlamaCpp for scenarios without a real model
@patch('core.llm_loader.LlamaCpp', autospec=True) # Patch LlamaCpp in the loader's namespace
def test_load_llm_mocked_success(mock_llama_cpp):
    """ Test loader logic with LlamaCpp mocked """
    mock_instance = MagicMock(spec=LlamaCpp)
    mock_instance.model_path = "mock/path/model.gguf"
    mock_llama_cpp.return_value = mock_instance

    # Set a dummy path in env to pass the existence check
    dummy_path = "dummy_model_path.gguf"
    set_key(dotenv_path, "MODEL_PATH", dummy_path)
    os.environ["MODEL_PATH"] = dummy_path
    # Create dummy file to pass os.path.exists check
    with open(dummy_path, "w") as f: f.write("dummy")

    llm = load_llm()

    # Assertions
    assert llm is mock_instance # Should return the mocked instance
    mock_llama_cpp.assert_called_once() # Check if LlamaCpp constructor was called
    # Check if it was called with the correct path
    args, kwargs = mock_llama_cpp.call_args
    assert kwargs.get('model_path') == dummy_path

    # Clean up dummy file
    os.remove(dummy_path) 