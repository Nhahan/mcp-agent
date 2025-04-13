import os
import logging
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List
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
    api_title: str = "MCP Agent API"
    api_description: str = "API for the MCP Agent using ReAct pattern and llama-cpp."
    api_version: str = "0.2.0"
    api_v1_prefix: str = "/api/v1" # API prefix
    
    # 모델 구성
    model_filename: str = "QwQ-LCoT-7B-Instruct-IQ4_NL.gguf" # Default model filename
    model_dir: Optional[str] = None # Directory for models, defaults below
    model_path: Optional[str] = None # Full path, overrides dir/filename if set
    n_ctx: int = 16384  # Reduced from 32768 to improve stability
    gpu_layers: int = -1 # -1 means offload all possible layers to GPU
    
    # LLM 생성 파라미터
    model_max_tokens: int = 1024
    model_temperature: float = 0.7
    model_top_p: float = 0.9
    model_top_k: int = 40
    model_min_p: float = 0.05 # Only used if model supports it

    # ReAct 루프 설정
    react_max_iterations: int = 10

    # 문법 파일 경로 (GBNF)
    grammar_path: str = "react_output.gbnf"  # Re-enable grammar for stable output

    # 로깅 및 디버깅
    log_level: str = "INFO"
    log_dir: Optional[str] = None # Log directory, defaults below
    
    # MCP 구성
    mcp_config_filename: str = "mcp.json"
    mcp_config_path: Optional[str] = None
    
    # Environment
    environment: str = "development" # 'development' or 'production'
    
    # Pydantic v2 경고 해결: protected_namespaces 설정 추가
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        protected_namespaces=('settings_',), # 'model_' 네임스페이스 충돌 방지
        extra='ignore' # 명시적으로 정의되지 않은 필드는 무시
    )
        
    def __init__(self, **values):
        super().__init__(**values)
        # Calculate default paths after loading from .env
        if self.model_dir is None:
            self.model_dir = str(PROJECT_ROOT / "models")
        if self.model_path is None:
             if self.model_filename:
                self.model_path = str(Path(self.model_dir) / self.model_filename)
             else:
                # Handle case where filename is also not provided (optional)
                self.model_path = None # Or raise an error
        
        if self.grammar_path is None:
            self.grammar_path = str(PROJECT_ROOT / "react_output.gbnf") # Default grammar file
            
        if self.mcp_config_path is None:
            self.mcp_config_path = str(PROJECT_ROOT / self.mcp_config_filename)
            
        if self.log_dir is None:
            self.log_dir = str(PROJECT_ROOT / "logs")
            
        # Ensure directories exist (optional, creates if not found)
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    @property
    def calculated_model_dir_path(self) -> Path:
        """Returns the model directory as a Path object."""
        return Path(self.model_dir)

    @property
    def calculated_model_path(self) -> Optional[Path]:
        """Calculates and returns the full model path as a Path object, or None."""
        # Use the property that returns a Path object for the directory check
        model_dir_p = self.calculated_model_dir_path 
        if not model_dir_p.exists():
            logger.warning(f"Model directory does not exist: {model_dir_p}")
            # Attempt to create it? Or return None?
            # model_dir_p.mkdir(parents=True, exist_ok=True)
            return None # Cannot determine path if directory doesn't exist

        if self.model_path:
            # If model_path is explicitly set, use it
            return Path(self.model_path)
        elif self.model_filename:
            # If only filename is set, construct path using the directory Path object
            return model_dir_p / self.model_filename
        else:
            # If neither path nor filename is set
            logger.warning("Neither model_path nor model_filename is configured.")
            return None

    @property
    def calculated_mcp_config_path(self) -> Path:
        """Returns the MCP config path as a Path object."""
        return Path(self.mcp_config_path)
        
    @property
    def calculated_log_dir_path(self) -> Path:
        """Returns the log directory as a Path object."""
        return Path(self.log_dir)

    @property
    def calculated_grammar_path(self) -> Optional[Path]:
        """Returns the grammar path as a Path object, or None."""
        if self.grammar_path:
             return Path(self.grammar_path)
        return None

# Function to get settings instance, cached for efficiency
@lru_cache()
def get_settings() -> Settings:
    logger.info("Loading application settings...")
    try:
        settings_instance = Settings()
        
        # Use the properties which return Path objects
        model_dir_path = settings_instance.calculated_model_dir_path
        model_path_obj = settings_instance.calculated_model_path
        mcp_config_path_obj = settings_instance.calculated_mcp_config_path
        log_dir_path = settings_instance.calculated_log_dir_path

        # 1. 모델 경로 유효성 검사 및 로깅
        if model_path_obj:
            logger.info(f"Using model path: {model_path_obj}")
            if not model_path_obj.exists():
                logger.warning(f"Model file ({model_path_obj}) does not exist yet. Ensure it's downloaded.")
        else:
             logger.error("Model path configuration is missing or invalid.")
             raise ValueError("Model path configuration could not be determined.")
             
        # 2. MCP 설정 파일 경로 유효성 검사
        logger.info(f"Using MCP config path: {mcp_config_path_obj}")
        if not mcp_config_path_obj.exists():
            logger.warning(f"MCP config file ({mcp_config_path_obj}) does not exist.")

        # 3. 로그 디렉토리 확인 (이미 __init__에서 생성 시도됨)
        logger.info(f"Using log directory: {log_dir_path}")
        if not log_dir_path.is_dir(): # Check if it's actually a directory
             logger.error(f"Log path ({log_dir_path}) exists but is not a directory.")
             # Or attempt to recreate/handle error

        logger.info("Settings loaded and paths validated successfully.")
        return settings_instance
    except Exception as e:
        logger.error(f"Fatal error loading settings: {e}", exc_info=True)
        raise

settings = get_settings()

# 애플리케이션 시작 시 주요 설정 로깅 (get_settings 내에서 이미 로깅하지만, 확인용으로 유지 가능)
logger.info("--- Application Configuration ---")
logger.info(f"API Title: {settings.api_title} v{settings.api_version}")
logger.info(f"Log Level: {settings.log_level}")
logger.info(f"Log Directory: {settings.calculated_log_dir_path}")
logger.info(f"Model Path: {settings.calculated_model_path}")
logger.info(f"Model Context (n_ctx): {settings.n_ctx}")
logger.info(f"GPU Layers: {settings.gpu_layers}")
logger.info(f"Grammar Path: {settings.calculated_grammar_path}")
logger.info(f"MCP Config Path: {settings.calculated_mcp_config_path}")
logger.info(f"ReAct Max Iterations: {settings.react_max_iterations}")
logger.info(f"Model Max Tokens: {settings.model_max_tokens}")
logger.info(f"Model Temperature: {settings.model_temperature}")
logger.info(f"Model Top-P: {settings.model_top_p}")
logger.info(f"Model Top-K: {settings.model_top_k}")
logger.info(f"Model Min-P: {settings.model_min_p}")
logger.info("-------------------------------")