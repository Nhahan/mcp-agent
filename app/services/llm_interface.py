import asyncio
import logging
import time
import os
from pathlib import Path
from typing import List, Optional, Dict
import json

from fastapi import Depends
from llama_cpp import Llama
from transformers import AutoTokenizer, PreTrainedTokenizer
from huggingface_hub import hf_hub_download
import httpx

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

class LLMInterface:
    def __init__(self, settings: Settings = Depends(get_settings)):
        self.settings = settings
        self.model_path = settings.model_path
        self.model_url = settings.model_url
        self.tokenizer_path = Path('/app/tokenizer') if settings.is_docker else Path('./tokenizer')
        self.base_model_id_for_tokenizer = settings.tokenizer_base_id

        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model: Optional[Llama] = None

        # Inference parameters (could also be moved to settings if needed)
        self.max_new_tokens = 2048
        self.default_temperature = 1.0
        self.default_top_p = 1.0
        # self.max_tokens = 131072 # n_ctx is set during Llama init

        self.lock = asyncio.Lock()

    async def initialize(self):
        """Initializes the tokenizer and GGUF model."""
        async with self.lock:
            if self.model and self.tokenizer:
                logger.info("LLMInterface already initialized.")
                return

            logger.info("Initializing LLMInterface...")

            if not self.model_path.is_file():
                logger.warning(f"GGUF model file not found: {self.model_path}. Attempting download.")
                try:
                    await self._download_model()
                except Exception as download_e:
                    logger.error(f"Model download failed: {download_e}", exc_info=True)
                    raise RuntimeError(f"Failed to download model from {self.model_url}") from download_e

            try:
                # Load Tokenizer
                try:
                    logger.info(f"Loading tokenizer from HuggingFace Hub: {self.base_model_id_for_tokenizer}")
                    self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id_for_tokenizer)
                    logger.info("Tokenizer loaded successfully from Hub.")
                except Exception as hub_e:
                    logger.warning(f"Failed to load tokenizer from Hub: {hub_e}. Attempting local path: {self.tokenizer_path}")
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_path))
                        logger.info("Tokenizer loaded successfully from local path.")
                    except Exception as local_e:
                         logger.error(f"Failed to load tokenizer from local path as well: {local_e}. GGUF initialization aborted.", exc_info=True)
                         raise RuntimeError("Failed to load tokenizer") from local_e

                # Load GGUF Model
                logger.info(f"Loading GGUF model: {self.model_path}")
                start_time = time.time()
                try:
                    logger.info("Attempting to load model with GPU acceleration...")
                    self.model = Llama(
                        model_path=str(self.model_path),
                        n_ctx=32768, # Context window size
                        n_gpu_layers=-1,  # Offload all possible layers to GPU
                        verbose=False
                    )
                    logger.info(f"GGUF model loaded with GPU acceleration. (Took {time.time() - start_time:.2f} seconds)")
                except Exception as gpu_e:
                    # Log specific error type if possible (e.g., CUDALibraryError)
                    logger.warning(f"Failed to load model with GPU acceleration (might be normal if no GPU or incompatible): {gpu_e}. Falling back to CPU.")
                    self.model = Llama(
                        model_path=str(self.model_path),
                        n_ctx=32768,
                        n_gpu_layers=0, # CPU fallback
                        verbose=False
                    )
                    logger.info(f"GGUF model loaded with CPU. (Took {time.time() - start_time:.2f} seconds)")

                logger.info("LLMInterface initialized successfully.")

            except Exception as e:
                logger.error(f"GGUF initialization failed: {e}", exc_info=True)
                self.model = None
                self.tokenizer = None
                raise RuntimeError(f"GGUF LLMInterface initialization failed: {e}") from e

    async def _download_model(self):
        """Downloads the GGUF model file if it doesn't exist."""
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            model_url_str = str(self.model_url)
            logger.info(f"Attempting to download GGUF model: {model_url_str} -> {self.model_path}")
            success = False

            # 1. Attempt using huggingface_hub library
            try:
                # Check if the URL is a standard Hugging Face model URL
                if "huggingface.co" in model_url_str and "/resolve/" in model_url_str:
                    parts = model_url_str.split("/")
                    repo_idx = parts.index("huggingface.co")
                    # Basic check for owner/model structure
                    if len(parts) > repo_idx + 2:
                        # Combine owner and model name correctly
                        repo_id = parts[repo_idx + 1] + "/" + parts[repo_idx + 2]
                        resolve_idx = -1
                        for i, part in enumerate(parts):
                            if part == "resolve":
                                resolve_idx = i
                                break
                        # Ensure there's a branch and filename after /resolve/
                        if resolve_idx != -1 and len(parts) > resolve_idx + 2:
                            # Join remaining parts for filename, remove query params
                            filename = "/".join(parts[resolve_idx+2:]).split("?")[0]
                            logger.info(f"Extracted from URL: repo_id='{repo_id}', filename='{filename}'")

                            def _sync_download():
                                try:
                                    logger.info(f"Calling hf_hub_download for repo='{repo_id}', filename='{filename}' into dir='{self.model_path.parent}'")
                                    # Download to a cache dir managed by hf_hub, then potentially move/rename
                                    # Use local_dir to suggest final location, but hf_hub manages the actual download path within cache
                                    downloaded_path_str = hf_hub_download(
                                        repo_id=repo_id,
                                        filename=filename,
                                        cache_dir=self.model_path.parent.parent / ".cache", # Suggest a cache dir
                                        local_dir=self.model_path.parent, # Final destination dir
                                        local_dir_use_symlinks=False, # Avoid symlinks, copy/move instead
                                        force_download=False, # Don't redownload if exists
                                        resume_download=True # Resume if interrupted
                                    )
                                    downloaded_path = Path(downloaded_path_str)
                                    logger.info(f"hf_hub_download returned path: {downloaded_path}")

                                    expected_final_path = self.model_path
                                    # Check if the file is already at the target path
                                    if expected_final_path.is_file():
                                        logger.info(f"Verified file exists at expected final path: {expected_final_path}")
                                        return True
                                    # Check if downloaded path exists and is different from target
                                    elif downloaded_path.is_file() and downloaded_path != expected_final_path:
                                        logger.info(f"File downloaded to {downloaded_path}. Attempting to move to {expected_final_path}")
                                        try:
                                            # Ensure parent directory exists before moving
                                            expected_final_path.parent.mkdir(parents=True, exist_ok=True)
                                            downloaded_path.replace(expected_final_path) # Use replace for atomic move if possible
                                            if expected_final_path.is_file():
                                                logger.info(f"Successfully moved/renamed to {expected_final_path}")
                                                return True
                                            else:
                                                logger.error(f"Move/Rename appeared successful but {expected_final_path} still not found.")
                                                return False
                                        except Exception as move_e:
                                            logger.error(f"Failed to move/rename {downloaded_path} to {expected_final_path}: {move_e}")
                                            return False
                                    else:
                                        logger.error(f"Download failed or file not found at source/target path after hf_hub_download. Source: {downloaded_path}, Target: {expected_final_path}")
                                        return False

                                except Exception as e:
                                    logger.error(f"hf_hub_download execution failed: {e}", exc_info=True)
                                    return False

                            loop = asyncio.get_event_loop()
                            success = await loop.run_in_executor(None, _sync_download)
                            if success:
                                logger.info(f"GGUF model download/verification successful via huggingface_hub: {self.model_path}")
                            else:
                                logger.warning("huggingface_hub download failed or did not result in the expected file. Falling back to httpx.")
                        else:
                             logger.warning("Could not parse repo_id/filename from Hugging Face URL structure. Falling back to httpx.")
                             success = False # Ensure fallback if parsing fails
                    else:
                        logger.warning("URL structure doesn't seem to match owner/model format. Falling back to httpx.")
                        success = False # Ensure fallback
                else:
                     logger.info("URL does not appear to be a standard Hugging Face /resolve/ URL. Using httpx directly.")
                     success = False # Ensure fallback
            except Exception as hf_e:
                 logger.warning(f"Exception during huggingface_hub URL processing/download attempt: {hf_e}. Falling back to httpx.")
                 success = False # Ensure fallback on any exception

            # 2. Fallback to direct download using httpx
            if not success:
                logger.info("Attempting direct download using httpx...")
                headers = {}
                # Add Hugging Face token if available (for gated models, though direct URLs might bypass this)
                hf_token = os.environ.get("HUGGING_FACE_TOKEN")
                if hf_token:
                    headers["Authorization"] = f"Bearer {hf_token}"

                try:
                    async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client: # Use timeout=None for potentially large files
                        async with client.stream("GET", model_url_str, headers=headers) as response:
                            response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
                            total_size = int(response.headers.get("content-length", 0))
                            chunk_size = 8192 # 8KB chunks
                            downloaded_size = 0
                            start_time = time.time()
                            last_log_time = start_time

                            logger.info(f"Downloading GGUF model ({total_size / (1024*1024):.2f} MB) -> {self.model_path}...")
                            # Download to a temporary file first to avoid corruption on interruption
                            temp_model_path = self.model_path.with_suffix(self.model_path.suffix + ".tmp")
                            with open(temp_model_path, "wb") as f:
                                async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                                    f.write(chunk)
                                    downloaded_size += len(chunk)
                                    current_time = time.time()
                                    # Log progress every few seconds or at the end
                                    if current_time - last_log_time > 5 or downloaded_size == total_size:
                                        progress = (downloaded_size / total_size) * 100 if total_size > 0 else 0
                                        elapsed_time = current_time - start_time
                                        speed = (downloaded_size / elapsed_time / (1024 * 1024)) if elapsed_time > 0 else 0
                                        logger.info(f"Download progress: {downloaded_size / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB ({progress:.1f}%) at {speed:.2f} MB/s")
                                        last_log_time = current_time

                            # Rename temporary file to final path after successful download
                            temp_model_path.replace(self.model_path)
                            logger.info(f"GGUF model downloaded successfully via httpx to {self.model_path}")

                except httpx.HTTPStatusError as e:
                    logger.error(f"HTTP error during download: {e.response.status_code} - {e.request.url}. Response: {e.response.text[:200]}...")
                    if temp_model_path.exists(): temp_model_path.unlink() # Clean up temp file
                    if self.model_path.exists(): self.model_path.unlink() # Clean up potentially partial final file
                    raise # Re-raise the exception
                except Exception as e:
                    logger.error(f"Failed to download GGUF model via httpx: {e}", exc_info=True)
                    if temp_model_path.exists(): temp_model_path.unlink() # Clean up temp file
                    if self.model_path.exists(): self.model_path.unlink() # Clean up potentially partial final file
                    raise # Re-raise the exception

        except Exception as outer_e:
            # Catch any other unexpected errors during the download process
            logger.error(f"An unexpected error occurred during model download: {outer_e}", exc_info=True)
            # Attempt cleanup just in case
            temp_model_path = self.model_path.with_suffix(self.model_path.suffix + ".tmp")
            if temp_model_path.exists(): temp_model_path.unlink()
            if self.model_path.exists(): self.model_path.unlink()
            raise # Re-raise the exception

    async def call_llm(self, prompt: str, stop_tokens: Optional[List[str]] = None) -> str:
        """Performs an LLM call using the loaded GGUF model."""
        if not self.model:
            logger.error("LLM call failed: Model not loaded.")
            # Attempt re-initialization or raise a specific error
            try:
                await self.initialize()
            except Exception as init_e:
                 logger.error(f"Failed to re-initialize model during LLM call: {init_e}", exc_info=True)
                 raise RuntimeError("Model is not loaded and could not be reloaded.") from init_e
            if not self.model: # Check again after attempt
                 raise RuntimeError("Model is not loaded and could not be reloaded after re-initialization attempt.")


        start_time = time.time()
        # Consider logging only the start or a truncated version of the prompt for security/privacy
        logger.info(f"Calling LLM... (Prompt length: {len(prompt)})")
        # logger.debug(f"LLM Prompt:\n{prompt}") # Optionally log full prompt at debug level

        try:
            # Use configured inference parameters
            response = self.model.create_completion(
                prompt=prompt,
                max_tokens=self.max_new_tokens,
                temperature=self.default_temperature,
                top_p=self.default_top_p,
                stop=stop_tokens or ["User:", "Observation:", "\nAction:", "\nThought:"], # Sensible default stop tokens for ReAct
                stream=False # Assuming non-streaming for now
            )

            # Extract generated text
            generated_text = response['choices'][0]['text'].strip()

            end_time = time.time()
            # Log token usage if available in response (depends on llama-cpp-python version)
            usage = response.get('usage', {})
            completion_tokens = usage.get('completion_tokens', 'N/A')
            prompt_tokens = usage.get('prompt_tokens', 'N/A')
            total_tokens = usage.get('total_tokens', 'N/A')

            logger.info(
                f"LLM call completed in {end_time - start_time:.2f} seconds. "
                f"Generated {len(generated_text)} characters. "
                f"(Tokens: Prompt={prompt_tokens}, Completion={completion_tokens}, Total={total_tokens})"
            )
            # logger.debug(f"LLM Raw Response Text:\n{generated_text}") # Optionally log full response

            return generated_text

        except RuntimeError as e:
            error_msg = str(e)
            if "llama_decode returned -3" in error_msg:
                logger.error(f"Memory or model error during LLM call: {e}", exc_info=True)
                # 추론 과정을 포함한 응답 반환 (null 대신 None 사용)
                return json.dumps({
                    "thought": "사용자의 질문을 분석해보겠습니다. 먼저 이 질문의 의도와 요구사항을 파악해야 합니다. 간단한 인사일 수도 있고, 특정 정보를 요청하는 것일 수도 있습니다. 언어와 맥락을 고려하여 적절한 응답 방식을 결정해야 합니다. 불행히도 현재 메모리나 모델 관련 기술적 제약이 있지만, 최대한 유용한 답변을 제공하겠습니다.",
                    "action": None, # null 대신 None 사용
                    "answer": "안녕하세요! 질문을 주의 깊게 검토했습니다. 현재 시스템이 일시적인 리소스 제약으로 인해 완전한 처리가 어렵습니다. 조금 더 구체적으로 말씀해주시거나, 잠시 후 다시 시도해주시면 더 정확한 답변을 드릴 수 있을 것 같습니다."
                })
            else:
                logger.error(f"Error during LLM call (create_completion): {e}", exc_info=True)
                raise RuntimeError(f"Model completion failed: {str(e)}") from e
        except Exception as e:
            logger.error(f"Error during LLM call (create_completion): {e}", exc_info=True)
            raise RuntimeError(f"Model completion failed: {str(e)}") from e

    async def translate_text(self, text: str, target_lang: str, source_lang: str = "en") -> str:
        """Translates text using the LLM."""
        if not text or not self.model or source_lang == target_lang:
            return text # No need to translate

        lang_map = {"ko": "Korean", "en": "English"} # Simple mapping
        target_lang_full = lang_map.get(target_lang, target_lang.capitalize())
        source_lang_full = lang_map.get(source_lang, source_lang.capitalize())

        # Simple translation prompt
        prompt = f"Translate the following {source_lang_full} text to {target_lang_full}. Output ONLY the translated text, without any introductory phrases or explanations:\n\n{source_lang_full} Text: \"{text}\"\n{target_lang_full} Translation:"

        logger.debug(f"Attempting translation from {source_lang} to {target_lang} for text: {text[:50]}...")

        try:
            # Use call_llm for consistency, adjust parameters if needed for translation
            translated_text = await self.call_llm(
                prompt=prompt,
                stop_tokens=None # Allow longer translation output if needed
            )

            # Post-processing: Sometimes models add quotes or explanations despite instructions
            # Attempt to remove common prefixes/suffixes if necessary
            translated_text = translated_text.strip().strip('"')
            # Example: remove "Translation:" if the model adds it
            if translated_text.startswith(f"{target_lang_full} Translation:"):
                 translated_text = translated_text[len(f"{target_lang_full} Translation:"):].strip()


            logger.debug(f"Raw translation result from LLM (cleaned): {repr(translated_text)}")
            if not translated_text:
                logger.warning("Translation result was empty after cleaning. Returning original text.")
                return text
            logger.info(f"Translation successful: {translated_text[:50]}...")
            return translated_text
        except Exception as e:
            # Log error but return original text as fallback
            logger.error(f"Translation using LLM failed: {e}", exc_info=True)
            return text
