import logging
import time
import os
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path

try:
    from llama_cpp import Llama  # Using llama-cpp-python
except ImportError:
    Llama = None
    logging.error("llama-cpp-python library not found. Please install it for LLM features.")

logger = logging.getLogger(__name__)

class ModelError(Exception):
    """Model-related errors"""
    pass

class ModelService:
    """모델 서비스 - 모델 초기화 및 관리를 담당"""
    
    def __init__(
        self,
        model_path: str,
        model_params: Dict[str, Any] = None,
    ):
        """
        ModelService 초기화
        
        Args:
            model_path: 모델 파일 경로
            model_params: 모델 초기화 매개변수
        """
        self.model = None
        self.model_path = model_path
        self.model_params = model_params or {}
        self.model_loaded = False
        
        logger.info("ModelService initialized with model path: %s", self.model_path)
        logger.info("Model parameters: %s", self.model_params)
        
        if Llama is None:
             logger.error("llama-cpp-python is not installed. LLM features will be disabled.")
    
    async def load_model(self) -> bool:
        """
        모델을 메모리에 로드합니다.
        
        Returns:
            성공 여부 (True/False)
        """
        logger.info("Starting model initialization...")
        try:
            from llama_cpp import Llama
            
            # 시간 측정 시작
            start_time = time.time()
            
            # 모델 파일 확인
            if not os.path.exists(self.model_path):
                error_msg = f"모델 파일을 찾을 수 없습니다: {self.model_path}"
                logger.critical(error_msg)
                raise ModelError(error_msg)
                
            # 모델 로드
            logger.info(f"모델 로드 시작. 경로: {self.model_path}, 매개변수: {self.model_params}")
            self.model = Llama(
                model_path=self.model_path,
                **self.model_params
            )
            
            # 소요 시간 계산 및 로깅
            elapsed_time = time.time() - start_time
            logger.info(f"모델 초기화 완료! 소요 시간: {elapsed_time:.2f}초")
            
            # 모델 로드 완료 플래그 설정
            self.model_loaded = True
            
            return True
        except Exception as e:
            error_msg = f"모델 초기화 중 오류 발생: {str(e)}"
            logger.critical(error_msg, exc_info=True)
            raise ModelError(error_msg) from e

    async def generate_chat(self, messages: list, **kwargs) -> str:
        """
        LLM을 사용하여 채팅 스타일의 텍스트를 생성합니다.
        
        Args:
            messages: 채팅 메시지 목록 (role, content 형식)
            **kwargs: 생성 매개변수 (temperature, max_tokens 등)
            
        Returns:
            생성된 텍스트
        """
        if not self.model_loaded or self.model is None:
            logger.error("LLM model is not loaded. Cannot generate text.")
            return "Error: LLM model is not available."

        try:
            # Set generation parameters
            temperature = kwargs.get("temperature", 0.3)
            max_tokens = kwargs.get("max_tokens", 4096)  # Default to 4K tokens
            top_p = kwargs.get("top_p", 0.9)
            top_k = kwargs.get("top_k", 40)
            repeat_penalty = kwargs.get("repeat_penalty", 1.1)
            
            # Log messages (sensitive information may be present, so log only a part)
            logger.debug(f"Generating chat completion for {len(messages)} messages")
            if messages:
                last_message = messages[-1].get('content', '')
                logger.debug(f"Last message (first 100 chars): {last_message[:100]}...")
            
            # Use chat-style API to generate text
            start_time = time.time()
            
            try:
                completion_result = await asyncio.to_thread(
                    self.model.create_chat_completion,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    stream=False
                )
                
                end_time = time.time()
                generation_time = end_time - start_time
                
                # Extract response
                if "choices" in completion_result and completion_result["choices"]:
                    response_text = completion_result["choices"][0]["message"]["content"]
                    logger.debug(f"Generated text in {generation_time:.2f}s: {response_text[:512]}...")
                    return response_text.strip()
                else:
                    logger.error(f"Unexpected response format from llama-cpp: {completion_result}")
                    return "Error: Unexpected response format from LLM."
                    
            except ValueError as e:
                if "exceed context window" in str(e):
                    logger.error(f"Context window exceeded: {e}")
                    return "Error: Context window size exceeded. Please try a shorter conversation."
                raise
                
        except Exception as e:
            logger.error(f"Error during chat generation: {e}", exc_info=True)
            return "Error: Could not generate chat response from LLM."
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """
        단일 프롬프트로 텍스트를 생성합니다.
        
        Args:
            prompt: 입력 프롬프트
            **kwargs: 생성 매개변수
            
        Returns:
            생성된 텍스트
        """
        # 프롬프트를 채팅 형식으로 변환
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        return await self.generate_chat(messages, **kwargs)
    
    async def shutdown(self) -> None:
        """모델 자원을 해제합니다."""
        logger.info("Shutting down model resources...")
        if self.model is not None:
            try:
                del self.model
            except Exception as e:
                logger.error(f"Error trying to delete model object: {e}")
            
            self.model = None
            self.model_loaded = False
            logger.info("Model resources released.")
            
        await asyncio.sleep(0) 