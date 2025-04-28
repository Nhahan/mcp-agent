import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.llms import LlamaCpp
import logging

# --- Logging Setup --- #
logger = logging.getLogger("LLM-Loader")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
# --- End Logging Setup --- #

load_dotenv()

# Default parameters
DEFAULT_N_CTX = 8192
DEFAULT_N_GPU_LAYERS = -1
DEFAULT_N_BATCH = 512
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 40
DEFAULT_VERBOSE = False

# --- LLM Singleton Loader --- #
llm_instance = None

def load_llm():
    global llm_instance
    if llm_instance is not None:
        logger.info("Returning existing LLM instance.")
        return llm_instance

    logger.info("Initializing new LLM instance...")
    model_path_str = os.getenv("MODEL_PATH")
    if not model_path_str:
        logger.error("MODEL_PATH environment variable not set.")
        raise ValueError("MODEL_PATH environment variable not set.")

    model_path = Path(model_path_str)
    if not model_path.exists():
        logger.error(f"Model file not found at path: {model_path}")
        raise FileNotFoundError(f"Model file not found at path: {model_path}")

    logger.info(f"Loading model: {model_path}")

    try:
        llm_instance = LlamaCpp(
            model_path=str(model_path),
            n_ctx=int(os.getenv("N_CTX", DEFAULT_N_CTX)),
            n_gpu_layers=int(os.getenv("N_GPU_LAYERS", DEFAULT_N_GPU_LAYERS)),
            n_batch=int(os.getenv("N_BATCH", DEFAULT_N_BATCH)),
            temperature=float(os.getenv("TEMPERATURE", DEFAULT_TEMPERATURE)),
            max_tokens=int(os.getenv("MAX_TOKENS", DEFAULT_MAX_TOKENS)),
            top_p=float(os.getenv("TOP_P", DEFAULT_TOP_P)),
            top_k=int(os.getenv("TOP_K", DEFAULT_TOP_K)),
            verbose=os.getenv("VERBOSE", str(DEFAULT_VERBOSE)).lower() == 'true',
            grammar=None, # Explicitly set grammar to None
        )
        logger.info("Model ready (Grammar constraints disabled).")
        return llm_instance
    except Exception as e:
        logger.error(f"Failed to load the LLM: {e}", exc_info=True)
        raise

# --- CLI Test Section --- #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("--- LLM Loader Test ---")
    try:
        llm = load_llm()
        logger.info("LLM loaded successfully via load_llm().")

    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Test failed: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during testing: {e}", exc_info=True)
# --- End CLI Test Section ---
