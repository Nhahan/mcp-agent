import os
from pathlib import Path
from dotenv import load_dotenv
import logging
from importlib import import_module # Added for dynamic imports
from typing import Optional, List # Added List

# --- LLM Classes --- #
from langchain_community.llms import LlamaCpp
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None # Handle optional import
try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None # Handle optional import

# --- Logging Setup --- #
logger = logging.getLogger(__name__) # Use __name__ for logger name
if not logger.handlers:
    # Configure root logger if no handlers are attached to this specific logger yet
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s')
# --- End Logging Setup --- #

load_dotenv()

# Default parameters for LlamaCpp
DEFAULT_N_CTX = 8192
DEFAULT_N_GPU_LAYERS = -1
DEFAULT_N_BATCH = 512

# Default parameters for API models (and LlamaCpp where applicable)
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 8192 # Increased default for API models
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 40
DEFAULT_VERBOSE = False # LlamaCpp specific, keep default False

# Default Model Names
DEFAULT_GEMINI_MODEL = "gemini-1.5-pro-latest"
DEFAULT_CLAUDE_MODEL = "claude-3-5-sonnet-20240620"

# --- LLM Singleton Loader --- #
llm_instance = None

def load_llm():
    global llm_instance
    if llm_instance is not None:
        logger.info("Returning existing LLM instance.")
        return llm_instance

    logger.info("Initializing new LLM instance...")

    # --- Check API Keys --- #
    google_api_key = os.getenv("GOOGLE_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    if google_api_key and anthropic_api_key:
        error_msg = "Both GOOGLE_API_KEY and ANTHROPIC_API_KEY are set. Please provide only one."
        logger.error(error_msg)
        raise ValueError(error_msg)

    # --- Load Gemini if API Key is present --- #
    if google_api_key:
        if ChatGoogleGenerativeAI is None:
            error_msg = "GOOGLE_API_KEY is set, but langchain-google-genai is not installed. Please install it."
            logger.error(error_msg)
            raise ImportError(error_msg)

        gemini_model_name = os.getenv("GEMINI_MODEL_NAME", DEFAULT_GEMINI_MODEL)
        logger.info(f"Using Gemini model: {gemini_model_name}")
        try:
            llm_instance = ChatGoogleGenerativeAI(
                model=gemini_model_name,
                google_api_key=google_api_key,
                temperature=float(os.getenv("TEMPERATURE", DEFAULT_TEMPERATURE)),
                max_output_tokens=int(os.getenv("MAX_TOKENS", DEFAULT_MAX_TOKENS)),
                top_p=float(os.getenv("TOP_P", DEFAULT_TOP_P)),
                top_k=int(os.getenv("TOP_K", DEFAULT_TOP_K)),
                # Add other relevant parameters if needed, checking ChatGoogleGenerativeAI documentation
            )
            logger.info("Google Gemini model loaded successfully.")
            return llm_instance
        except Exception as e:
            logger.error(f"Failed to load Google Gemini model: {e}", exc_info=True)
            raise

    # --- Load Claude if API Key is present --- #
    elif anthropic_api_key:
        if ChatAnthropic is None:
            error_msg = "ANTHROPIC_API_KEY is set, but langchain-anthropic is not installed. Please install it."
            logger.error(error_msg)
            raise ImportError(error_msg)

        claude_model_name = os.getenv("CLAUDE_MODEL_NAME", DEFAULT_CLAUDE_MODEL)
        logger.info(f"Using Claude model: {claude_model_name}")
        try:
            llm_instance = ChatAnthropic(
                model=claude_model_name,
                anthropic_api_key=anthropic_api_key,
                temperature=float(os.getenv("TEMPERATURE", DEFAULT_TEMPERATURE)),
                max_tokens=int(os.getenv("MAX_TOKENS", DEFAULT_MAX_TOKENS)),
                top_p=float(os.getenv("TOP_P", DEFAULT_TOP_P)),
                top_k=int(os.getenv("TOP_K", DEFAULT_TOP_K)),
                # Add other relevant parameters if needed, checking ChatAnthropic documentation
            )
            logger.info("Anthropic Claude model loaded successfully.")
            return llm_instance
        except Exception as e:
            logger.error(f"Failed to load Anthropic Claude model: {e}", exc_info=True)
            raise

    # --- Fallback to LlamaCpp if no API keys are present --- #
    else:
        logger.info("No API keys found. Attempting to load local LlamaCpp model...")
        model_path_str = os.getenv("MODEL_PATH")
        if not model_path_str:
            error_msg = "No API keys (GOOGLE_API_KEY, ANTHROPIC_API_KEY) or MODEL_PATH environment variable set. Cannot load LLM."
            logger.error(error_msg)
            raise ValueError(error_msg)

        model_path = Path(model_path_str)
        if not model_path.exists():
            error_msg = f"MODEL_PATH specified, but model file not found at path: {model_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info(f"Loading local LlamaCpp model: {model_path}")

        # --- Load LlamaCpp Instance --- #
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
            )
            logger.info("LlamaCpp base instance ready.")
            return llm_instance
        except Exception as e:
            logger.error(f"Failed to load the LlamaCpp LLM: {e}", exc_info=True)
            raise

# --- Grammar Generation Helper REMOVED --- #

# --- CLI Test Section --- #
if __name__ == "__main__":
    # Ensure test execution uses the updated logic
    logging.basicConfig(level=logging.DEBUG) # Use DEBUG for more test info
    logger = logging.getLogger(__name__) # Re-get logger after basicConfig potentially called by root
    logger.info("--- LLM Loader Test ---")

    # --- GBNF Test REMOVED --- #

    # Clear instance for LLM load test
    llm_instance = None
    logger.info("Testing LLM loading...")
    # Test scenarios (Uncomment or set env vars locally to test each case)
    # os.environ["GOOGLE_API_KEY"] = "test-key" # Test Gemini
    # os.environ["ANTHROPIC_API_KEY"] = "test-key" # Test Claude
    # os.environ["MODEL_PATH"] = "models/gemma-3-12b-it-qat-int4-Q4_K_M.gguf" # Test LlamaCpp (ensure path is correct)
    # os.environ["GOOGLE_API_KEY"] = "test-key" # Test conflict
    # os.environ["ANTHROPIC_API_KEY"] = "test-key"

    try:
        # Set a default model path if none is set for testing LlamaCpp loading path
        if not os.getenv("GOOGLE_API_KEY") and not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("MODEL_PATH"):
             logger.warning("Neither API key nor MODEL_PATH set. Skipping LLM load test.")
        else:
            llm = load_llm()
            logger.info(f"LLM loaded successfully via load_llm(). Type: {type(llm)}")

    except (ValueError, FileNotFoundError, ImportError) as e:
        logger.error(f"LLM load test failed as expected or due to config issue: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during LLM loading test: {e}", exc_info=True)
# --- End CLI Test Section ---
