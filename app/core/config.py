import os
import logging
from pathlib import Path
from functools import lru_cache
from pydantic import Field, HttpUrl, AliasChoices, DirectoryPath
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Determine project root based on this file's location
# This assumes config.py is in app/core/
PROJECT_ROOT = Path(__file__).parent.parent.parent

class Settings(BaseSettings):
    # API 구성
    api_title: str = "MCP Agent"
    api_description: str = "MCP Agent API"
    api_version: str = "0.1.0"
    
    # 모델 구성
    model_path: Path = Path("test_model_cache/gemma-3-1b-it-q4_0.gguf")
    model_url: HttpUrl = "https://huggingface.co/google/gemma-3-1b-it-qat-q4_0-gguf/resolve/main/gemma-3-1b-it-q4_0.gguf?download=true"
    model_repo_id: str = "google/gemma-3-1b-it-qat-q4_0-gguf"  # Hugging Face 리포지토리 ID
    model_filename: str = "gemma-3-1b-it-q4_0.gguf"  # 파일 이름
    
    # 로깅 및 디버깅
    log_level: str = "INFO"
    
    # MCP 구성
    mcp_config_path: Path = Path("mcp.json")
    
    # Docker 환경 체크 
    is_docker: bool = False  # Docker 환경인지 확인 (파일 경로에 영향)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        
        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            return (
                init_settings,
                env_settings,
                _docker_check,
                file_secret_settings,
            )
    
    # 유틸리티 속성들
    @property
    def MODEL_CACHE_DIR(self) -> Path:
        """모델 캐시 디렉토리 반환"""
        cache_dir = self.model_path.parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    @property
    def MCP_CONFIG_PATH(self) -> Path:
        """MCP 설정 파일 경로 반환 (대문자 속성명 - 하위 호환성)"""
        return self.mcp_config_path
    
    @property
    def model_path_str(self) -> str:
        """모델 경로를 문자열로 반환"""
        return str(self.model_path)
    
    @property
    def mcp_config_path_str(self) -> str:
        """MCP 설정 경로를 문자열로 반환"""
        return str(self.mcp_config_path)

def _docker_check(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Docker 환경인지 확인합니다."""
    # /.dockerenv 파일이 존재하는지 확인
    settings["is_docker"] = Path("/.dockerenv").exists()
    return settings

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
        logger.debug(f"MCP Config Path: {settings.mcp_config_path_str}")
        logger.debug(f"Log Level: {settings.log_level}") # Log the level
        return settings
    except Exception as e:
        logger.error(f"Error loading settings: {e}", exc_info=True)
        raise

# Remove the direct instantiation at module level
# settings = Settings() 