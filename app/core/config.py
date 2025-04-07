import os
import logging
from pathlib import Path
from functools import lru_cache
from pydantic import Field, HttpUrl, AliasChoices, DirectoryPath
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, Any
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Determine project root based on this file's location
# This assumes config.py is in app/core/
PROJECT_ROOT = Path(__file__).parent.parent.parent

# .env 파일 로딩 시도 (선택적)
load_dotenv()

class Settings(BaseSettings):
    # API 구성
    api_title: str = "MCP Agent"
    api_description: str = "MCP Agent API"
    api_version: str = "0.1.0"
    
    # 모델 구성 (정확한 모델 정보로 업데이트)
    model_repo_id: str = os.getenv("MODEL_REPO_ID", "mradermacher/gemma3-4b-it-abliterated-GGUF")
    model_filename: str = os.getenv("MODEL_FILENAME", "gemma3-4b-it-abliterated.Q8_0.gguf")
    tokenizer_base_id: str = os.getenv("TOKENIZER_BASE_ID", "google/gemma-3-4b-it")
    model_dir: Path = Path('/app/models') if os.getenv("RUNNING_IN_DOCKER", "False").lower() == "true" else Path('./models')
    
    # 로깅 및 디버깅
    log_level: str = "INFO"
    
    # MCP 구성
    mcp_config_path: Path = Path('/app/mcp.json') if os.getenv("RUNNING_IN_DOCKER", "False").lower() == "true" else Path('./mcp.json')
    
    # Hugging Face 토큰
    hugging_face_token: str = ""
    
    # Docker 환경 체크 
    is_docker: bool = os.getenv("RUNNING_IN_DOCKER", "False").lower() == "true"
    
    # Pydantic v2 경고 해결: protected_namespaces 설정 추가
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        protected_namespaces=('settings_',), # 'model_' 네임스페이스 충돌 방지
        extra='ignore' # 명시적으로 정의되지 않은 필드는 무시 (기본값)
    )
        
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
    
    # 유틸리티 속성들 (URL 자동 생성)
    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_filename
        
    @property
    def model_url(self) -> str:
        """Constructs the model download URL from repo_id and filename."""
        return f"https://huggingface.co/{self.model_repo_id}/resolve/main/{self.model_filename}"

    @property
    def MCP_CONFIG_PATH(self) -> Path:
        """MCP 설정 파일 경로 반환"""
        return self.mcp_config_path

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
        return settings
    except Exception as e:
        logger.error(f"Error loading settings: {e}", exc_info=True)
        raise

# settings 인스턴스를 여기서 생성합니다.
settings = get_settings()

# 애플리케이션 시작 시 모델 디렉토리 생성
if not settings.model_dir.exists():
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created model directory: {settings.model_dir}")

# 시작 시 주요 설정 로깅 (model_url 속성 사용)
logger.info(f"Model Repo ID set to: {settings.model_repo_id}")
logger.info(f"Model Filename set to: {settings.model_filename}")
logger.info(f"Tokenizer Base ID set to: {settings.tokenizer_base_id}")
logger.info(f"Calculated Model URL: {settings.model_url}")
logger.info(f"Model path set to: {settings.model_path}")
logger.info(f"MCP config path: {settings.mcp_config_path}")
logger.info(f"Running in Docker: {settings.is_docker}") 