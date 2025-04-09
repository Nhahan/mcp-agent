import os
import logging
from pathlib import Path
from functools import lru_cache
from pydantic import Field, HttpUrl, AliasChoices, DirectoryPath
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, Any, Optional
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
    
    # 모델 구성
    model_filename: str = os.getenv("MODEL_FILENAME", "")
    model_dir: Path = Path('/app/models') if os.getenv("RUNNING_IN_DOCKER", "False").lower() == "true" else Path('./models')
    
    # 로깅 및 디버깅
    log_level: str = "INFO"
    log_dir: str = "logs"
    
    # MCP 구성
    mcp_config_path: str = "mcp.json"
    
    model_path: Optional[str] = None # .env 에서 읽어올 모델 파일 경로
    n_ctx: int = 32768 # 모델 컨텍스트 길이
    gpu_layers: int = -1 # GPU에 오프로드할 레이어 수 (0이면 CPU만 사용)
    
    # Pydantic v2 경고 해결: protected_namespaces 설정 추가
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        protected_namespaces=('settings_',), # 'model_' 네임스페이스 충돌 방지
        extra='ignore' # 명시적으로 정의되지 않은 필드는 무시 (기본값)
    )
        
    @property
    def calculated_model_path(self) -> Path:
        """모델 디렉토리와 파일 이름을 조합하여 전체 모델 경로를 반환합니다."""
        # 모델 디렉토리가 존재하지 않으면 생성
        if not self.model_dir.exists():
            try:
                self.model_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create model directory {self.model_dir}: {e}")
                # 디렉토리 생성 실패 시 기본 경로 반환 또는 예외 발생 선택
                # 여기서는 일단 파일 이름만 있는 Path 객체 반환 (오류 유발 가능성 있음)
                return Path(self.model_filename)
        return self.model_dir / self.model_filename

# Function to get settings instance, cached for efficiency
@lru_cache()
def get_settings() -> Settings:
    logger.debug("Loading application settings...")
    try:
        settings_instance = Settings()
        
        # model_path가 .env에 명시적으로 설정되었는지 확인
        model_path_from_env = os.getenv("MODEL_PATH")
        if model_path_from_env:
             settings_instance.model_path = model_path_from_env # .env 값 사용
             logger.info(f"Using MODEL_PATH from environment: {settings_instance.model_path}")
             # 파일 존재 여부 체크 (다운로드 전일 수 있음)
             if not Path(settings_instance.model_path).exists():
                   logger.error(f"MODEL_PATH file ({settings_instance.model_path}) does not exist yet.")
                   raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {settings_instance.model_path}. 애플리케이션을 실행하기 전에 모델 파일을 다운로드하세요.")
        else:
             # .env에 없으면 계산된 경로를 사용하도록 설정 (None 유지 시 문제 발생 가능)
             calculated = settings_instance.calculated_model_path
             settings_instance.model_path = str(calculated)
             logger.info(f"MODEL_PATH not set in environment. Using calculated path: {settings_instance.model_path}")
             # 계산된 경로의 파일 존재 여부 체크
             if not calculated.exists():
                   logger.error(f"Calculated model path file ({calculated}) does not exist yet.")
                   raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {calculated}. 애플리케이션을 실행하기 전에 모델 파일을 다운로드하세요.")
                   
                   
        # mcp_config_path도 Path 객체로 변환 (필요시)
        if isinstance(settings_instance.mcp_config_path, str):
             settings_instance.mcp_config_path = Path(settings_instance.mcp_config_path)

        return settings_instance
    except Exception as e:
        logger.error(f"Error loading settings: {e}", exc_info=True)
        raise

# settings 인스턴스를 여기서 생성합니다.
settings = get_settings()

# 애플리케이션 시작 시 모델 디렉토리 생성
if not settings.model_dir.exists():
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created model directory: {settings.model_dir}")

# 시작 시 주요 설정 로깅
logger.info(f"Model Filename set to: {settings.model_filename}")
logger.info(f"Model path set to: {settings.model_path}")
logger.info(f"MCP config path: {settings.mcp_config_path}")