import os
import logging
from pathlib import Path
from functools import lru_cache
from pydantic import Field, HttpUrl, AliasChoices, DirectoryPath
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Determine project root based on this file's location
# This assumes config.py is in app/core/
PROJECT_ROOT = Path(__file__).parent.parent.parent

class Settings(BaseSettings):
    # Define fields and explicitly state how they map to env vars using validation_alias
    model_url: HttpUrl = Field(
        ...,
        validation_alias=AliasChoices('model_url', 'MODEL_URL'),
        description="URL to download the ONNX model"
    )
    model_path: Path = Field(
        default=PROJECT_ROOT / "test_model_cache" / "model_q4f16.onnx",
        validation_alias=AliasChoices('model_path', 'MODEL_PATH'),
        description="Path to store/load the ONNX model"
    )
    mcp_config_path: Path = Field(
        default=PROJECT_ROOT / "mcp.json",
        validation_alias=AliasChoices('mcp_config_path', 'MCP_CONFIG_PATH'),
        description="Path to the MCP configuration file"
    )
    log_level: str = Field(
        "INFO",
        validation_alias=AliasChoices('log_level', 'LOG_LEVEL'),
        description="Logging level"
    )

    # pydantic-settings v2 configuration
    model_config = SettingsConfigDict(
        env_prefix='',
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
        # case_sensitive=False # Alternatively, set case_sensitive globally
    )

    # Derived paths
    @property
    def model_path_str(self) -> str:
        return str(self.model_path)

    @property
    def mcp_config_path_str(self) -> str:
        return str(self.mcp_config_path)

    @property
    def MCP_CONFIG_PATH(self) -> Path:
        # Ensure the path is resolved relative to the project root if it's relative
        path = self.mcp_config_path
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    @property
    def MODEL_CACHE_DIR(self) -> Path:
        # Ensure the parent directory exists
        cache_dir = self.model_path.parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

# Function to get settings instance, cached for efficiency
@lru_cache()
def get_settings() -> Settings:
    logger.debug("Loading application settings...")
    try:
        settings = Settings()
        # Log crucial paths after loading
        logger.debug(f"Model URL: {settings.model_url}")
        logger.debug(f"Model Path: {settings.model_path}")
        logger.debug(f"Model Cache Dir: {settings.MODEL_CACHE_DIR}")
        logger.debug(f"MCP Config Path: {settings.MCP_CONFIG_PATH}")
        logger.debug(f"Log Level: {settings.log_level}") # Log the level
        return settings
    except Exception as e:
        logger.error(f"Error loading settings: {e}", exc_info=True)
        raise

# Remove the direct instantiation at module level
# settings = Settings() 