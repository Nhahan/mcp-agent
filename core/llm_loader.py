import os
from pathlib import Path
from dotenv import load_dotenv
import logging

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
try:
    # Use the community OpenAI integration which is more up-to-date
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None # Handle optional import for OpenRouter/OpenAI

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
# DEFAULT_TOP_K = 40 # Removed as often not supported
DEFAULT_VERBOSE = False # LlamaCpp specific, keep default False

# Default model names are removed as MODEL_NAME is now mandatory for APIs
# DEFAULT_GEMINI_MODEL = "gemini-1.5-pro-latest"
# DEFAULT_CLAUDE_MODEL = "claude-3-5-sonnet-20240620"
# DEFAULT_OPEN_ROUTER_MODEL = "qwen/qwen3-14b:free"

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
    open_router_api_key = os.getenv("OPEN_ROUTER_API_KEY")

    # --- Get Provider Hint and MODEL_NAME --- #
    llm_provider = os.getenv("LLM_PROVIDER", "").lower()
    # MODEL_NAME is now the ONLY source for the model ID for API providers
    env_model_name = os.getenv("MODEL_NAME")

    # --- Determine which LLM to load based on keys and explicit provider setting --- #

    # 1. OpenRouter Check (Requires Key AND MODEL_NAME)
    # Use if explicit provider hint OR only OR key is set
    if open_router_api_key and (
        llm_provider == 'openrouter'
        or not (google_api_key or anthropic_api_key) # Implicit selection if only OR key exists
    ):
        if not env_model_name:
             error_msg = "OPEN_ROUTER_API_KEY is set, but MODEL_NAME environment variable is missing. Cannot load OpenRouter LLM without a specified model name."
             logger.error(error_msg)
             raise ValueError(error_msg)
        if ChatOpenAI is None:
            error_msg = "OPEN_ROUTER_API_KEY is set, but langchain-openai is not installed..." # Truncated for brevity
            logger.error(error_msg)
            raise ImportError(error_msg)

        if not (llm_provider == 'openrouter'):
             logger.info(f"Only OPEN_ROUTER_API_KEY found. Implicitly selecting OpenRouter.")
        logger.info(f"Using OpenRouter model from MODEL_NAME: {env_model_name}")

        try:
            model_kwargs = {
                "top_p": float(os.getenv("TOP_P", DEFAULT_TOP_P))
            }
            llm_instance = ChatOpenAI(
                model=env_model_name, # Directly use MODEL_NAME
                openai_api_key=open_router_api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=float(os.getenv("TEMPERATURE", DEFAULT_TEMPERATURE)),
                max_tokens=int(os.getenv("MAX_TOKENS", DEFAULT_MAX_TOKENS)),
                model_kwargs=model_kwargs
            )
            logger.info("OpenRouter model loaded successfully via ChatOpenAI.")
            return llm_instance
        except Exception as e:
            logger.error(f"Failed to load OpenRouter model: {e}", exc_info=True)
            raise

    # 2. Gemini Check (Requires Key AND MODEL_NAME)
    # Use if explicit provider hint OR only Google key is set
    elif google_api_key and (llm_provider == 'google' or not (anthropic_api_key or open_router_api_key)):
        if not env_model_name:
             error_msg = "GOOGLE_API_KEY is set, but MODEL_NAME environment variable is missing. Cannot load Gemini LLM without a specified model name."
             logger.error(error_msg)
             raise ValueError(error_msg)
        if ChatGoogleGenerativeAI is None:
            error_msg = "GOOGLE_API_KEY is set, but langchain-google-genai is not installed..." # Truncated
            logger.error(error_msg)
            raise ImportError(error_msg)

        if not (llm_provider == 'google'):
            logger.info(f"Only GOOGLE_API_KEY found. Implicitly selecting Google Gemini.")
        logger.info(f"Using Gemini model from MODEL_NAME: {env_model_name}")
        try:
            llm_instance = ChatGoogleGenerativeAI(
                model=env_model_name, # Directly use MODEL_NAME
                google_api_key=google_api_key,
                temperature=float(os.getenv("TEMPERATURE", DEFAULT_TEMPERATURE)),
                max_output_tokens=int(os.getenv("MAX_TOKENS", DEFAULT_MAX_TOKENS)),
                top_p=float(os.getenv("TOP_P", DEFAULT_TOP_P)),
                top_k=int(os.getenv("TOP_K", 40)) # Keep TOP_K for Gemini
            )
            logger.info("Google Gemini model loaded successfully.")
            return llm_instance
        except Exception as e:
            logger.error(f"Failed to load Google Gemini model: {e}", exc_info=True)
            raise

    # 3. Claude Check (Requires Key AND MODEL_NAME)
    # Use if explicit provider hint OR only Anthropic key is set
    elif anthropic_api_key and (llm_provider == 'anthropic' or not (google_api_key or open_router_api_key)):
        if not env_model_name:
             error_msg = "ANTHROPIC_API_KEY is set, but MODEL_NAME environment variable is missing. Cannot load Claude LLM without a specified model name."
             logger.error(error_msg)
             raise ValueError(error_msg)
        if ChatAnthropic is None:
            error_msg = "ANTHROPIC_API_KEY is set, but langchain-anthropic is not installed..." # Truncated
            logger.error(error_msg)
            raise ImportError(error_msg)

        if not (llm_provider == 'anthropic'):
            logger.info(f"Only ANTHROPIC_API_KEY found. Implicitly selecting Anthropic Claude.")
        logger.info(f"Using Claude model from MODEL_NAME: {env_model_name}")
        try:
            llm_instance = ChatAnthropic(
                model=env_model_name, # Directly use MODEL_NAME
                anthropic_api_key=anthropic_api_key,
                temperature=float(os.getenv("TEMPERATURE", DEFAULT_TEMPERATURE)),
                max_tokens=int(os.getenv("MAX_TOKENS", DEFAULT_MAX_TOKENS)),
                top_p=float(os.getenv("TOP_P", DEFAULT_TOP_P)),
                top_k=int(os.getenv("TOP_K", 40)) # Keep TOP_K for Claude
            )
            logger.info("Anthropic Claude model loaded successfully.")
            return llm_instance
        except Exception as e:
            logger.error(f"Failed to load Anthropic Claude model: {e}", exc_info=True)
            raise

    # 4. LlamaCpp Check (Requires MODEL_PATH)
    # Use if explicit provider hint OR no API keys set
    elif llm_provider == 'local' or not (google_api_key or anthropic_api_key or open_router_api_key):
        logger.info("No API keys specified or 'local' provider requested. Attempting to load local LlamaCpp model...")
        model_path_str = os.getenv("MODEL_PATH")
        if not model_path_str:
            error_msg = "LLM_PROVIDER is 'local' or no API keys provided, but MODEL_PATH environment variable is not set. Cannot load local LLM."
            logger.error(error_msg)
            raise ValueError(error_msg)

        model_path = Path(model_path_str)
        if not model_path.exists():
            error_msg = f"MODEL_PATH specified, but model file not found at path: {model_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info(f"Loading local LlamaCpp model: {model_path}")
        try:
            llm_instance = LlamaCpp(
                model_path=str(model_path),
                n_ctx=int(os.getenv("N_CTX", DEFAULT_N_CTX)),
                n_gpu_layers=int(os.getenv("N_GPU_LAYERS", DEFAULT_N_GPU_LAYERS)),
                n_batch=int(os.getenv("N_BATCH", DEFAULT_N_BATCH)),
                temperature=float(os.getenv("TEMPERATURE", DEFAULT_TEMPERATURE)),
                max_tokens=int(os.getenv("MAX_TOKENS", DEFAULT_MAX_TOKENS)),
                top_p=float(os.getenv("TOP_P", DEFAULT_TOP_P)),
                top_k=int(os.getenv("TOP_K", 40)), # Keep TOP_K for LlamaCpp
                verbose=os.getenv("VERBOSE", str(DEFAULT_VERBOSE)).lower() == 'true',
            )
            logger.info("LlamaCpp base instance ready.")
            return llm_instance
        except Exception as e:
            logger.error(f"Failed to load the LlamaCpp LLM: {e}", exc_info=True)
            raise

    # 5. Error: Ambiguous or Incomplete Configuration
    else:
         # This case now primarily catches situations where multiple API keys are set
         # without an explicit LLM_PROVIDER hint, OR if an API key is set but MODEL_NAME is missing.
         error_msg = "Could not determine which LLM to load. Check API keys (GOOGLE_API_KEY, ANTHROPIC_API_KEY, OPEN_ROUTER_API_KEY) and ensure only one provider is implicitly or explicitly selected. If using an API provider, ensure MODEL_NAME environment variable is also set."
         logger.error(error_msg)
         if google_api_key: logger.error(" - GOOGLE_API_KEY is set.")
         if anthropic_api_key: logger.error(" - ANTHROPIC_API_KEY is set.")
         if open_router_api_key: logger.error(" - OPEN_ROUTER_API_KEY is set.")
         if not env_model_name and (google_api_key or anthropic_api_key or open_router_api_key):
             logger.error(" - MODEL_NAME is MISSING but an API key is set.")
         raise ValueError(error_msg)

# --- Cleanup Function (Optional) --- #
def cleanup_llm():
    """Explicitly clears the global LLM instance."""
    global llm_instance
    if llm_instance:
        logger.info("Cleaning up LLM instance.")
        # Add any specific cleanup logic if needed (e.g., closing connections)
        llm_instance = None
    else:
        logger.info("No LLM instance to clean up.")
