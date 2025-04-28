import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.llms import LlamaCpp
from llama_cpp import LlamaGrammar # Keep this import just in case
import logging

# Removed dynamic grammar generation imports and logic
# try:
#     from pydantic_gbnf_grammar_generator import generate_gbnf_grammar_and_documentation
#     PYDANTIC_GBNF_AVAILABLE = True
# except ImportError:
#     PYDANTIC_GBNF_AVAILABLE = False
#     generate_gbnf_grammar_and_documentation = None
#     logger = logging.getLogger("LLM-Loader") # Initialize logger early for warning
#     logger.warning("pydantic-gbnf-grammar-generator not found. Proceeding without dynamic grammar generation.")

# Removed model import as it's no longer needed for grammar generation here
# from core.schemas.plan_output import PlanOutput 

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
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 40
DEFAULT_VERBOSE = False

# --- LLM Singleton Loader --- #
llm_instance = None
# Removed grammar instance caching
# grammar_instance = None 

def load_llm():
    global llm_instance
    if llm_instance is not None:
        logger.info("Returning existing LLM instance.")
        return llm_instance

    # --- Grammar generation logic removed --- 
    # grammar = None
    # if PYDANTIC_GBNF_AVAILABLE:
    #     logger.info("Attempting to generate GBNF grammar dynamically from PlanOutput model...")
    #     try:
    #         grammar_str, _ = generate_gbnf_grammar_and_documentation([PlanOutput])
    #         grammar = LlamaGrammar.from_string(grammar_str)
    #         logger.info("GBNF grammar generated dynamically and loaded successfully.")
    #     except Exception as e:
    #         logger.error(f"Failed to generate or load dynamic grammar: {e}. Proceeding without grammar.", exc_info=True)
    #         grammar = None
    # else:
    #     logger.warning("Cannot generate grammar because pydantic-gbnf-grammar-generator is not installed.")
    #     grammar = None
    # --- End Grammar Generation --- #

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

        # Removed grammar check as grammar is disabled
        # if grammar_instance:
        #     logger.info("Dynamically generated grammar object is present in the loaded LLM instance.")
        # else:
        #     logger.warning("No grammar object found or generated for the LLM instance.")
        logger.info("Grammar constraints are disabled for this LLM instance.")

    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Test failed: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during testing: {e}", exc_info=True)
# --- End CLI Test Section ---
