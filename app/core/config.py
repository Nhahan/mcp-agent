import os
import logging
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from functools import lru_cache
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
    model_path: Optional[str] = os.getenv("MODEL_PATH") # .env 에서 읽어올 모델 파일 경로, get_settings에서 최종 결정됨
    n_ctx: int = int(os.getenv("N_CTX", 32768))
    gpu_layers: int = int(os.getenv("GPU_LAYERS", -1))
    
    # LLM 생성 파라미터
    model_max_tokens: int = int(os.getenv("MODEL_MAX_TOKENS", 8192))
    model_temperature: float = float(os.getenv("MODEL_TEMPERATURE", 0.6))
    model_top_p: float = float(os.getenv("MODEL_TOP_P", 0.95))
    model_top_k: int = int(os.getenv("MODEL_TOP_K", 40))
    model_min_p: float = float(os.getenv("MODEL_MIN_P", 0))

    # ReAct 루프 설정
    react_max_iterations: int = int(os.getenv("MAX_ITERATIONS", 10)) # 최대 반복 횟수

    # 문법 파일 경로 (GBNF)
    grammar_path: str = os.getenv("GRAMMAR_PATH", "react_output.gbnf") # .env 또는 기본값 사용

    # 로깅 및 디버깅
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_dir: str = os.getenv("LOG_DIR", "logs")
    
    # MCP 구성
    mcp_config_filename: str = os.getenv("MCP_CONFIG_FILENAME", "mcp.json")
    mcp_config_path: Optional[Path] = None # get_settings에서 최종 결정됨
    
    # Pydantic v2 경고 해결: protected_namespaces 설정 추가
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        protected_namespaces=('settings_',), # 'model_' 네임스페이스 충돌 방지
        extra='ignore' # 명시적으로 정의되지 않은 필드는 무시
    )
        
    @property
    def calculated_model_path(self) -> Path:
        """모델 디렉토리와 파일 이름을 조합하여 전체 모델 경로를 반환합니다."""
        if not self.model_dir.exists():
            try:
                self.model_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created model directory: {self.model_dir}")
            except Exception as e:
                logger.error(f"Failed to create model directory {self.model_dir}: {e}", exc_info=True)
                # 디렉토리 생성 실패 시에도 일단 경로 객체 반환 시도
                return Path(self.model_filename) # 잠재적 오류 가능
        return self.model_dir / self.model_filename

    @property
    def calculated_mcp_config_path(self) -> Path:
        """프로젝트 루트와 MCP 설정 파일 이름을 조합하여 전체 경로를 반환합니다."""
        return PROJECT_ROOT / self.mcp_config_filename
        
# Function to get settings instance, cached for efficiency
@lru_cache()
def get_settings() -> Settings:
    logger.info("Loading application settings...")
    try:
        settings_instance = Settings()
        
        # 1. 모델 경로 결정 (MODEL_PATH 환경변수 우선)
        model_path_from_env = os.getenv("MODEL_PATH")
        if model_path_from_env:
            settings_instance.model_path = model_path_from_env
            logger.info(f"Using MODEL_PATH from environment: {settings_instance.model_path}")
            # 파일 존재 여부 체크 (시작 시)
            if not Path(settings_instance.model_path).exists():
                logger.warning(f"MODEL_PATH file ({settings_instance.model_path}) does not exist yet. Ensure it's downloaded before use.")
                # 여기서 에러를 발생시킬 수도 있지만, 다운로드 로직이 별도로 있다면 경고만 남길 수 있음
                # raise FileNotFoundError(f"Model file does not exist: {settings_instance.model_path}. Download the model file before running the application.")
        elif settings_instance.model_filename:
            # 환경변수 없고 filename 있으면 계산된 경로 사용
            calculated = settings_instance.calculated_model_path
            settings_instance.model_path = str(calculated)
            logger.info(f"MODEL_PATH not set. Using calculated path based on MODEL_FILENAME: {settings_instance.model_path}")
            # 계산된 경로 파일 존재 여부 체크 (시작 시)
            if not calculated.exists():
                logger.warning(f"Calculated model path file ({calculated}) does not exist yet. Ensure it's downloaded before use.")
        else:
            # 둘 다 없으면 오류
             logger.error("Neither MODEL_PATH environment variable nor MODEL_FILENAME is set. Cannot determine model path.")
             raise ValueError("Model path configuration is missing. Set either MODEL_PATH or MODEL_FILENAME.")
             
        # 2. MCP 설정 파일 경로 결정
        settings_instance.mcp_config_path = settings_instance.calculated_mcp_config_path
        logger.info(f"Using MCP config path: {settings_instance.mcp_config_path}")
        if not settings_instance.mcp_config_path.exists():
            logger.warning(f"MCP config file ({settings_instance.mcp_config_path}) does not exist.")
            # mcp.json이 필수적인 경우 여기서 에러 발생
            # raise FileNotFoundError(f"MCP config file not found: {settings_instance.mcp_config_path}")

        # 3. 로그 디렉토리 생성
        log_path = Path(settings_instance.log_dir)
        if not log_path.exists():
            try:
                log_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created log directory: {log_path}")
            except Exception as e:
                logger.error(f"Failed to create log directory {log_path}: {e}", exc_info=True)

        logger.info("Settings loaded successfully.")
        return settings_instance
    except Exception as e:
        logger.error(f"Fatal error loading settings: {e}", exc_info=True)
        raise

settings = get_settings()

# 애플리케이션 시작 시 주요 설정 로깅 (get_settings 내에서 이미 로깅하지만, 확인용으로 유지 가능)
logger.info("--- Application Configuration ---")
logger.info(f"API Title: {settings.api_title} v{settings.api_version}")
logger.info(f"Log Level: {settings.log_level}")
logger.info(f"Log Directory: {settings.log_dir}")
logger.info(f"Model Path: {settings.model_path}")
logger.info(f"Model Context (n_ctx): {settings.n_ctx}")
logger.info(f"GPU Layers: {settings.gpu_layers}")
logger.info(f"Grammar Path: {settings.grammar_path}")
logger.info(f"MCP Config Path: {settings.mcp_config_path}")
logger.info(f"ReAct Max Iterations: {settings.react_max_iterations}")
logger.info(f"Model Max Tokens: {settings.model_max_tokens}")
logger.info(f"Model Temperature: {settings.model_temperature}")
logger.info(f"Model Top-P: {settings.model_top_p}")
logger.info(f"Model Top-K: {settings.model_top_k}")
logger.info(f"Model Min-P: {settings.model_min_p}")
logger.info("-------------------------------")