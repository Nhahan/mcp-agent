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

DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 8192 # Increased default for API models
DEFAULT_TOP_P = 0.95
# DEFAULT_TOP_K is often not supported by OpenAI/OpenRouter compatible APIs
# DEFAULT_TOP_K = 40
DEFAULT_VERBOSE = False # LlamaCpp specific, keep default False

# Default Model Names
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash-lite"
DEFAULT_CLAUDE_MODEL = "claude-3-5-sonnet-20240620"
DEFAULT_OPEN_ROUTER_MODEL = "qwen/qwen3-32b:free"

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

    # --- Prioritize Environment Variable for Provider/Model --- #
    llm_provider = os.getenv("LLM_PROVIDER", "").lower()
    model_name = os.getenv("MODEL_NAME")

    # --- Determine which LLM to load based on keys and explicit provider setting --- #

    # 1. OpenRouter Check (Explicit Hint OR Only OpenRouter Key Set)
    if open_router_api_key and (
        llm_provider == 'openrouter'
        or (model_name and model_name.startswith("openrouter/"))
        or not (google_api_key or anthropic_api_key) # Implicit: Use if only OpenRouter key exists
    ):
        if ChatOpenAI is None:
            error_msg = "OPEN_ROUTER_API_KEY is set, but langchain-openai is not installed. Please install it (`pip install langchain-openai`)."
            logger.error(error_msg)
            raise ImportError(error_msg)

        # Determine model name: Use explicit MODEL_NAME if it starts with openrouter/, else default
        open_router_model = model_name if (model_name and model_name.startswith("openrouter/")) else DEFAULT_OPEN_ROUTER_MODEL
        # Log if implicit selection is happening
        if not (llm_provider == 'openrouter' or (model_name and model_name.startswith("openrouter/"))):
            logger.info(f"Only OPEN_ROUTER_API_KEY found. Implicitly selecting OpenRouter.")
        logger.info(f"Using OpenRouter model: {open_router_model}")

        try:
            model_kwargs = {
                "top_p": float(os.getenv("TOP_P", DEFAULT_TOP_P))
            }
            
            llm_instance = ChatOpenAI(
                model=open_router_model,
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

    # 2. Gemini Check (Explicit Hint OR Only Google Key Set)
    elif google_api_key and (llm_provider == 'google' or not (anthropic_api_key or open_router_api_key)):
        if ChatGoogleGenerativeAI is None:
            error_msg = "GOOGLE_API_KEY is set, but langchain-google-genai is not installed. Please install it."
            logger.error(error_msg)
            raise ImportError(error_msg)

        gemini_model_name = model_name if model_name else DEFAULT_GEMINI_MODEL
        if not (llm_provider == 'google'):
            logger.info(f"Only GOOGLE_API_KEY found. Implicitly selecting Google Gemini.")
        logger.info(f"Using Gemini model: {gemini_model_name}")
        try:
            # Keep top_k for Gemini as it supports it
            llm_instance = ChatGoogleGenerativeAI(
                model=gemini_model_name,
                google_api_key=google_api_key,
                temperature=float(os.getenv("TEMPERATURE", DEFAULT_TEMPERATURE)),
                max_output_tokens=int(os.getenv("MAX_TOKENS", DEFAULT_MAX_TOKENS)),
                top_p=float(os.getenv("TOP_P", DEFAULT_TOP_P)),
                top_k=int(os.getenv("TOP_K", 40)) # Pass default TOP_K if env var not set
            )
            logger.info("Google Gemini model loaded successfully.")
            return llm_instance
        except Exception as e:
            logger.error(f"Failed to load Google Gemini model: {e}", exc_info=True)
            raise

    # 3. Claude Check (Explicit Hint OR Only Anthropic Key Set)
    elif anthropic_api_key and (llm_provider == 'anthropic' or not (google_api_key or open_router_api_key)):
        if ChatAnthropic is None:
            error_msg = "ANTHROPIC_API_KEY is set, but langchain-anthropic is not installed. Please install it."
            logger.error(error_msg)
            raise ImportError(error_msg)

        claude_model_name = model_name if model_name else DEFAULT_CLAUDE_MODEL
        if not (llm_provider == 'anthropic'):
            logger.info(f"Only ANTHROPIC_API_KEY found. Implicitly selecting Anthropic Claude.")
        logger.info(f"Using Claude model: {claude_model_name}")
        try:
            # Keep top_k for Claude as it supports it
            llm_instance = ChatAnthropic(
                model=claude_model_name,
                anthropic_api_key=anthropic_api_key,
                temperature=float(os.getenv("TEMPERATURE", DEFAULT_TEMPERATURE)),
                max_tokens=int(os.getenv("MAX_TOKENS", DEFAULT_MAX_TOKENS)),
                top_p=float(os.getenv("TOP_P", DEFAULT_TOP_P)),
                top_k=int(os.getenv("TOP_K", 40)) # Pass default TOP_K if env var not set
            )
            logger.info("Anthropic Claude model loaded successfully.")
            return llm_instance
        except Exception as e:
            logger.error(f"Failed to load Anthropic Claude model: {e}", exc_info=True)
            raise

    # 4. LlamaCpp Check (Explicit Hint OR No API Keys Set)
    elif llm_provider == 'local' or not (google_api_key or anthropic_api_key or open_router_api_key):
        logger.info("No API keys specified or 'local' provider requested. Attempting to load local LlamaCpp model...")
        model_path_str = os.getenv("MODEL_PATH")
        if not model_path_str:
            error_msg = "LLM_PROVIDER is 'local' or no API keys provided, but MODEL_PATH environment variable is not set. Cannot load LLM."
            logger.error(error_msg)
            raise ValueError(error_msg)

        model_path = Path(model_path_str)
        if not model_path.exists():
            error_msg = f"MODEL_PATH specified, but model file not found at path: {model_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info(f"Loading local LlamaCpp model: {model_path}")
        try:
            # Keep top_k for LlamaCpp as it supports it
            llm_instance = LlamaCpp(
                model_path=str(model_path),
                n_ctx=int(os.getenv("N_CTX", DEFAULT_N_CTX)),
                n_gpu_layers=int(os.getenv("N_GPU_LAYERS", DEFAULT_N_GPU_LAYERS)),
                n_batch=int(os.getenv("N_BATCH", DEFAULT_N_BATCH)),
                temperature=float(os.getenv("TEMPERATURE", DEFAULT_TEMPERATURE)),
                max_tokens=int(os.getenv("MAX_TOKENS", DEFAULT_MAX_TOKENS)),
                top_p=float(os.getenv("TOP_P", DEFAULT_TOP_P)),
                top_k=int(os.getenv("TOP_K", 40)), # Pass default TOP_K if env var not set
                verbose=os.getenv("VERBOSE", str(DEFAULT_VERBOSE)).lower() == 'true',
            )
            logger.info("LlamaCpp base instance ready.")
            return llm_instance
        except Exception as e:
            logger.error(f"Failed to load the LlamaCpp LLM: {e}", exc_info=True)
            raise

    # 5. Error: Ambiguous Configuration (Multiple Keys Set without Explicit Hint)
    else:
         error_msg = "Could not determine which LLM to load. Multiple API keys (GOOGLE_API_KEY, ANTHROPIC_API_KEY, OPEN_ROUTER_API_KEY) may be set without an explicit provider selection via LLM_PROVIDER or a specific MODEL_NAME format."
         logger.error(error_msg)
         if google_api_key: logger.error(" - GOOGLE_API_KEY is set.")
         if anthropic_api_key: logger.error(" - ANTHROPIC_API_KEY is set.")
         if open_router_api_key: logger.error(" - OPEN_ROUTER_API_KEY is set.")
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
