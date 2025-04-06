import asyncio
import logging
import re # Import regex module
from pathlib import Path
import httpx
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
from fastapi import Depends

from app.core.config import Settings, get_settings
from app.services.mcp_service import MCPService # Import MCPService

logger = logging.getLogger(__name__)

# Define the base model identifier for tokenizer loading
BASE_MODEL_ID = "google/gemma-3-1b-it"

class InferenceService:
    def __init__(self, settings: Settings = Depends(get_settings), mcp_service: MCPService = Depends()): # Inject MCPService
        self.settings = settings
        self.mcp_service = mcp_service # Store MCPService instance
        self.model_path = settings.model_path
        self.model_url = settings.model_url # Still HttpUrl type here
        self.tokenizer = None
        self.session = None
        self.lock = asyncio.Lock()

        self._download_task = None
        # Regex to detect command execution requests
        self.terminal_command_pattern = re.compile(r"(?:터미널에서|terminal에서|run command) (.*)(?: 실행| run)", re.IGNORECASE)
        # Store model metadata like num_layers if needed for past_key_values
        self.num_layers = 26 # Adjust based on the actual gemma-3-1b model config if different
        # Update num_heads and head_dim based on ONNXRuntime error expectation
        self.num_heads = 1
        self.head_dim = 256
        self.kv_dtype = np.float32

    async def initialize(self):
        async with self.lock:
            if self.session and self.tokenizer:
                logger.info("Model and tokenizer already initialized.")
                return

            if not self.model_path.is_file():
                if self._download_task is None:
                    logger.info(f"Model not found at {self.model_path}. Starting download...")
                    self._download_task = asyncio.create_task(self._download_model())
                else:
                    logger.info("Model download already in progress...")
                # Wait for download task to complete
                await self._download_task
                self._download_task = None # Reset task after completion
                if not self.model_path.is_file():
                    logger.error("Model download failed or file still not found.")
                    raise RuntimeError("Model download failed.")

            try:
                # Load tokenizer directly from the base model identifier on Hugging Face Hub
                logger.info(f"Loading tokenizer for base model: {BASE_MODEL_ID}...")
                self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
                logger.info("Tokenizer loaded successfully.")

                # Load ONNX session from the downloaded file
                logger.info(f"Loading ONNX model session from {self.model_path}...")
                sess_options = ort.SessionOptions()
                available_providers = ort.get_available_providers()
                logger.info(f"Available ORT Providers: {available_providers}")
                providers = []
                if 'CoreMLExecutionProvider' in available_providers:
                    logger.info("Using CoreMLExecutionProvider")
                    providers.append('CoreMLExecutionProvider')
                elif 'CUDAExecutionProvider' in available_providers:
                     logger.info("Using CUDAExecutionProvider")
                     providers.append('CUDAExecutionProvider')
                else:
                     logger.info("Using CPUExecutionProvider")
                     providers.append('CPUExecutionProvider')
                self.session = ort.InferenceSession(str(self.model_path), sess_options=sess_options, providers=providers)
                logger.info("ONNX model session loaded successfully.")
                logger.info("Inference service initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize tokenizer or model session: {e}", exc_info=True)
                self.session = None
                self.tokenizer = None
                raise RuntimeError(f"Failed to initialize inference service: {e}") from e

    async def _download_model(self):
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            # Convert Pydantic HttpUrl to string for httpx
            model_url_str = str(self.model_url)
            logger.info(f"Attempting to download from URL (as string): {model_url_str}")
            async with httpx.AsyncClient(follow_redirects=True) as client:
                # Use the string representation of the URL
                async with client.stream("GET", model_url_str) as response:
                    response.raise_for_status() # Raise exception for bad status codes
                    total_size = int(response.headers.get("content-length", 0))
                    chunk_size = 8192
                    downloaded_size = 0
                    logger.info(f"Downloading model ({total_size / (1024*1024):.2f} MB) to {self.model_path}...")
                    with open(self.model_path, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            # Log progress occasionally
                            if downloaded_size % (chunk_size * 128) == 0 or downloaded_size == total_size: # Log roughly every MB
                                progress = (downloaded_size / total_size) * 100 if total_size else 0
                                logger.info(f"Download progress: {downloaded_size / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB ({progress:.1f}%)")
            logger.info("Model download completed.")
        except Exception as e:
            logger.error(f"Failed to download model: {e}", exc_info=True)
            # Clean up potentially corrupted file
            if self.model_path.exists():
                self.model_path.unlink()
            # Re-raise to indicate failure
            raise

    async def generate(self, text: str) -> str:
        logger.info(f"지시: {text}") # Log the input instruction
        final_response = "" # Variable to hold the final response for logging
        try:
            async with self.lock:
                if not self.session or not self.tokenizer:
                    await self.initialize()
                    if not self.session or not self.tokenizer:
                        raise RuntimeError("Model not initialized. Download might be in progress or failed.")

                # 1. Check for Terminal Command Pattern
                match = self.terminal_command_pattern.search(text)
                if match and "iterm-mcp" in self.mcp_service.get_running_servers():
                    command_to_run = match.group(1).strip()
                    logger.info(f"Detected terminal command request: '{command_to_run}'")
                    try:
                        # Call write_to_terminal via MCPService
                        logger.debug(f"Calling write_to_terminal with: {command_to_run}")
                        await self.mcp_service.call_mcp_tool("iterm-mcp", "write_to_terminal", {"command": command_to_run})
                        await asyncio.sleep(1) # Give terminal time to execute

                        # Call read_terminal_output via MCPService
                        logger.debug("Calling read_terminal_output")
                        read_result = await self.mcp_service.call_mcp_tool("iterm-mcp", "read_terminal_output", {"linesOfOutput": 10})

                        # Extract and return terminal output
                        if read_result and isinstance(read_result.get("content"), list) and len(read_result["content"]) > 0:
                            terminal_output = read_result["content"][0].get("text", "")
                            logger.info("Returning terminal output.")
                            final_response = f"Terminal Output:\n```\n{terminal_output}\n```"
                        else:
                            logger.warning("Could not read terminal output after command execution.")
                            final_response = "명령어를 실행했지만 터미널 출력을 읽지 못했습니다."
                    except Exception as e:
                        logger.error(f"Error executing terminal command via MCP: {e}", exc_info=True)
                        final_response = f"터미널 명령어 실행 중 오류 발생: {e}"
                    
                    logger.info(f"응답: {final_response}") # Log the MCP response
                    return final_response

                # 2. If no command pattern, proceed with normal inference
                logger.info(f"Generating text for prompt: {text[:50]}...")
                inputs = self.tokenizer(text, return_tensors="np", return_attention_mask=False)

                # Prepare position_ids (essential for Gemma)
                input_ids = inputs["input_ids"]
                batch_size, sequence_length = input_ids.shape
                position_ids = ort.OrtValue.ortvalue_from_numpy(
                    np.arange(0, sequence_length, dtype=np.int64).reshape(batch_size, sequence_length)
                )
                
                # Initialize ort_inputs dictionary *before* the loop
                ort_inputs = {
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                }

                # Create dummy past_key_values for the first inference step
                # Shape: [batch_size, num_key_value_heads, sequence_length, head_dim]
                # For the *first* call, sequence_length is 0 for past_kv
                # Use the stored model config values (adjust if necessary)
                # Note: Shape might differ slightly based on specific ONNX export settings
                past_kv_shape = (batch_size, self.num_heads, 0, self.head_dim)
                input_names = [inp.name for inp in self.session.get_inputs()]
                for i in range(self.num_layers):
                    key_name = f'past_key_values.{i}.key'
                    value_name = f'past_key_values.{i}.value'
                    if key_name in input_names:
                        ort_inputs[key_name] = np.zeros(past_kv_shape, dtype=self.kv_dtype)
                        logger.debug(f"Adding dummy input: {key_name} with shape {past_kv_shape}")
                    if value_name in input_names:
                        ort_inputs[value_name] = np.zeros(past_kv_shape, dtype=self.kv_dtype)
                        logger.debug(f"Adding dummy input: {value_name} with shape {past_kv_shape}")

                # Run inference
                try:
                    logger.debug(f"Running inference with inputs: {list(ort_inputs.keys())}")
                    outputs = self.session.run(None, ort_inputs)
                    generated_ids_raw = outputs[0] # Shape: (batch_size, sequence_length)
                    logger.debug(f"Raw generated_ids type: {type(generated_ids_raw)}, shape: {getattr(generated_ids_raw, 'shape', 'N/A')}")
                    logger.debug(f"Raw generated_ids content (first 10): {generated_ids_raw[0][:10]}")
                    
                    # Decode directly from the numpy array slice
                    token_ids_to_decode = generated_ids_raw[0]
                    logger.debug(f"Token IDs to decode type: {type(token_ids_to_decode)}, first 10: {token_ids_to_decode[:10]}")

                    full_generated_text = self.tokenizer.decode(token_ids_to_decode, skip_special_tokens=True)
                    logger.info("Text generation successful.")
                    if full_generated_text.startswith(text):
                        final_response = full_generated_text[len(text):].strip()
                    else:
                        final_response = full_generated_text # Fallback
                except Exception as e:
                    logger.error(f"Error during model inference: {e}", exc_info=True)
                    raise RuntimeError(f"Model inference failed: {e}") from e

        except Exception as e:
            # Catch potential errors in the outer block (like initialization failure)
            logger.error(f"Error in generate function: {e}", exc_info=True)
            final_response = f"An internal error occurred: {e}" # Provide error info

        logger.info(f"응답: {final_response}") # Log the final response before returning
        return final_response

# Dependency function
def get_inference_service(settings: Settings = Depends(get_settings), mcp_service: MCPService = Depends()) -> InferenceService:
    # This pattern allows InferenceService to be treated like a singleton
    # within the application context if managed correctly (e.g., by FastAPI's dependency injection)
    # However, creating it here means a new instance per request unless FastAPI optimizes it.
    # For true singleton behavior, manage it at app startup.
    # For now, this ensures dependencies are injected.
    # Consider adding caching or app-level state management if initialization is expensive.
    return InferenceService(settings=settings, mcp_service=mcp_service) 