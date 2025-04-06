import os
import pytest
from pydantic import ValidationError, HttpUrl

# Make sure tests can import from the 'app' directory
# This might require setting PYTHONPATH or using pytest fixtures

# Test basic settings loading with environment variables

def test_settings_load_success(monkeypatch):
    """Test successful loading of Settings from environment variables."""
    monkeypatch.setenv("MODEL_URL", "http://example.com/model.onnx")
    monkeypatch.setenv("MODEL_PATH", "/path/to/model.onnx")
    monkeypatch.setenv("MCP_CONFIG_PATH", "/path/to/mcp.json")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    from app.core.config import get_settings, Settings
    get_settings.cache_clear()
    settings = Settings(_env_file=None)

    # Compare HttpUrl by converting it to string
    assert str(settings.model_url) == "http://example.com/model.onnx"
    # Or compare against a pre-constructed HttpUrl object
    # assert settings.model_url == HttpUrl("http://example.com/model.onnx")
    assert str(settings.model_path) == "/path/to/model.onnx"
    assert str(settings.mcp_config_path) == "/path/to/mcp.json"
    assert settings.log_level == "DEBUG"

def test_settings_load_missing_required(monkeypatch):
    """Test failure when a required environment variable is missing."""
    monkeypatch.setenv("MODEL_URL", "http://example.com/model.onnx")
    # MODEL_PATH is missing
    monkeypatch.setenv("MCP_CONFIG_PATH", "/path/to/mcp.json")
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False) # Also ensure log_level isn't set

    from app.core.config import get_settings, Settings
    get_settings.cache_clear()

    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)

    errors = excinfo.value.errors(include_url=False)
    assert len(errors) == 1
    assert errors[0]["loc"] == ("model_path_str",)
    assert errors[0]["type"] == "missing"

def test_settings_default_log_level(monkeypatch):
    """Test that LOG_LEVEL defaults to INFO if not set."""
    monkeypatch.setenv("MODEL_URL", "http://example.com/model.onnx")
    monkeypatch.setenv("MODEL_PATH", "/path/to/model.onnx")
    monkeypatch.setenv("MCP_CONFIG_PATH", "/path/to/mcp.json")
    monkeypatch.delenv("LOG_LEVEL", raising=False)

    from app.core.config import get_settings, Settings
    get_settings.cache_clear()
    settings = Settings(_env_file=None)
    assert settings.log_level == "INFO" 