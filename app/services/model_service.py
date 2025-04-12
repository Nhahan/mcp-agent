import logging
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path

try:
    from llama_cpp import Llama, LlamaGrammar
except ImportError:
    Llama = None
    LlamaGrammar = None
    logging.error("llama-cpp-python library not found. Please install it for LLM features.")

from app.core.config import settings

logger = logging.getLogger(__name__)

class ModelError(Exception):
    """Model-related errors"""
    pass

class ModelService:
    """모델 서비스 - 모델 초기화 및 관리를 담당"""
    
    def __init__(
        self,
        # model_path and model_params will now be primarily determined by settings
        # They can be overridden via arguments if needed, but default to settings
        model_path: Optional[str] = None,
        model_params: Optional[Dict[str, Any]] = None
    ):
        """
        ModelService 초기화. 설정은 주로 app.core.config.settings 에서 가져옵니다.
        
        Args:
            model_path: 모델 파일 경로 (지정되지 않으면 settings.model_path 사용).
            model_params: 모델 초기화 매개변수 (지정되지 않으면 settings 에서 n_ctx, n_gpu_layers 사용).
        """
        # Use provided path/params or fallback to settings
        self.model_path = model_path if model_path else settings.model_path
        
        if not self.model_path:
            raise ModelError("Model path is not configured in settings or provided to ModelService.")
        
        # Construct model_params from settings if not provided
        if model_params is None:
            self.model_params = {
                "n_ctx": settings.n_ctx, 
                "n_gpu_layers": settings.gpu_layers
                # Add other llama-cpp specific init params from settings if needed
            }
            logger.info(f"Using model params from settings: {self.model_params}")
        else:
            self.model_params = model_params
            logger.info(f"Using provided model params: {self.model_params}")
            
        # Ensure required params are present
        self.model_context_limit = self.model_params.get("n_ctx", settings.n_ctx) # Fallback to settings default again
        self.n_gpu_layers = self.model_params.get("n_gpu_layers", settings.gpu_layers)
        
        self.model: Optional[Llama] = None
        self.model_loaded = False
        self.grammar: Optional[LlamaGrammar] = None

        if not Path(self.model_path).exists():
            logger.error(f"Model file not found at {self.model_path}. Ensure it is downloaded.")
            # Consider not raising here, but checking model_loaded before use
            # raise ModelError(f"Model file not found: {self.model_path}")
            self.model_loaded = False # Explicitly set to false
            return # Exit init if model file doesn't exist

        # Load Grammar from settings.grammar_path
        grammar_path_str = settings.grammar_path
        if grammar_path_str:
            grammar_path = Path(grammar_path_str)
            if grammar_path.is_file(): # Check if it's a file
                try:
                    self.grammar = LlamaGrammar.from_file(str(grammar_path))
                    logger.info(f"Successfully loaded grammar from {grammar_path}")
                except Exception as e:
                    logger.error(f"Failed to load or parse grammar file {grammar_path}: {e}", exc_info=True)
                    self.grammar = None
            else:
                logger.warning(f"Grammar path specified ({grammar_path}), but it's not a file or doesn't exist.")
                self.grammar = None
        else:
            logger.info("No grammar path specified in settings. Grammar will not be used.")
            self.grammar = None
        
        # Load the LLM model
        try:
            if Llama is None:
                raise ModelError("llama-cpp-python is not installed.")
                
            logger.info(f"Loading LLM model from: {self.model_path} with n_ctx={self.model_context_limit}, n_gpu_layers={self.n_gpu_layers}")
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.model_context_limit, 
                n_gpu_layers=self.n_gpu_layers,
                verbose=settings.log_level.upper() == "DEBUG", # Be verbose only in debug mode
            )
            self.model_loaded = True
            logger.info("LLM model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load LLM model from {self.model_path}: {e}", exc_info=True)
            self.model_loaded = False

    async def generate_chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        generation_params = {
            "max_tokens": kwargs.get("max_tokens", settings.model_max_tokens),
            "temperature": kwargs.get("temperature", settings.model_temperature),
            "top_p": kwargs.get("top_p", settings.model_top_p),
            "top_k": kwargs.get("top_k", settings.model_top_k),
            "min_p": kwargs.get("min_p", settings.model_min_p),
            "stop": kwargs.get("stop", ["</s>", "<|eot_id|>", "<|endoftext|>", "\nObservation:", "\nUSER:", "```"]),
        }
        
        # Apply grammar ONLY if it's loaded AND not explicitly disabled in kwargs
        if "grammar" in kwargs:
                # Allow explicit override (e.g., grammar=None for translation)
                if kwargs["grammar"] is not None:
                    generation_params["grammar"] = kwargs["grammar"]
                    logger.debug("Applying grammar provided in kwargs.")
                else:
                    logger.debug("Grammar explicitly disabled by kwargs=None.")
        elif self.grammar: # Otherwise, apply the default loaded grammar if it exists
                generation_params["grammar"] = self.grammar
                logger.debug("Applying default loaded grammar.")
        final_params = generation_params

        output: Dict[str, Any] = await asyncio.to_thread(
                 self.model.create_chat_completion, messages=messages, **final_params
        )
        
        # Extract generated text
        if output and 'choices' in output and output['choices']:
                generated_text = output['choices'][0].get('message', {}).get('content', '').strip()
        else:
                logger.warning(f"Unexpected output structure from create_chat_completion: {output}")
                generated_text = ""
        return generated_text

    async def generate_text(self, prompt: str, **kwargs) -> str:
        """
        단일 프롬프트로 텍스트를 생성합니다. (generate_chat 사용)
        
        Args:
            prompt: 입력 프롬프트
            **kwargs: 생성 매개변수 (generate_chat으로 전달됨)
            
        Returns:
            생성된 텍스트
        """
        messages = [
            # Basic system prompt, can be made configurable if needed
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        # Pass kwargs directly to generate_chat, allowing override of defaults
        return await self.generate_chat(messages, **kwargs)
    
    async def shutdown(self) -> None:
        """모델 자원을 해제합니다."""
        logger.info("Shutting down model resources...")
        if self.model is not None:
            # llama-cpp doesn't have an explicit close/shutdown method
            # Deleting the object should release resources via __del__
            try:
                del self.model # Rely on garbage collection and __del__
            except Exception as e:
                logger.error(f"Error trying to delete model object: {e}")
            
            self.model = None
            self.model_loaded = False
            logger.info("Model resources marked for release.")
            
        await asyncio.sleep(0) # Yield control briefly 