from typing import Optional
import logging

# 필요한 서비스 클래스 임포트 (실제 경로 확인 필요)
try:
    from app.services.mcp_service import MCPService
    from app.services.inference_service import InferenceService
except ImportError as e:
     # 임포트 실패 시 로깅 (애플리케이션 시작 시 오류 확인 가능)
     logging.getLogger(__name__).error(f"Failed to import services in dependencies.py: {e}")
     MCPService = None
     InferenceService = None

# --- 싱글톤 인스턴스 저장 변수 ---
_mcp_service_instance: Optional[MCPService] = None
_inference_service_instance: Optional[InferenceService] = None

def set_mcp_service_instance(instance: MCPService):
    """Sets the global MCPService instance."""
    global _mcp_service_instance
    _mcp_service_instance = instance

def set_inference_service_instance(instance: InferenceService):
    """Sets the global InferenceService instance."""
    global _inference_service_instance
    _inference_service_instance = instance

# --- 의존성 주입 함수 (Getter) ---
def get_mcp_service() -> MCPService:
     """Returns the singleton MCPService instance."""
     if _mcp_service_instance is None:
         # 이 오류는 lifespan 이 올바르게 설정 및 실행되지 않았음을 의미
         raise RuntimeError("MCPService has not been initialized. Ensure the application lifespan context is correctly set up.")
     return _mcp_service_instance

def get_inference_service() -> InferenceService:
     """Returns the singleton InferenceService instance."""
     if _inference_service_instance is None:
         # 이 오류는 lifespan 이 올바르게 설정 및 실행되지 않았음을 의미
         raise RuntimeError("InferenceService has not been initialized. Ensure the application lifespan context is correctly set up.")
     return _inference_service_instance 