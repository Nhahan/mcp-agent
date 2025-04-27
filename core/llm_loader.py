# core/llm_loader.py
import os
from dotenv import load_dotenv
# from llama_cpp import Llama # No longer needed directly
from langchain_community.llms import LlamaCpp # Import Langchain wrapper
from typing import Optional

# Load environment variables from .env file
load_dotenv()

# Global variable to hold the loaded Langchain LLM instance
llm_instance: Optional[LlamaCpp] = None

# Define default parameters for LlamaCpp - consider moving to config or env vars
DEFAULT_LLAMA_PARAMS = {
    "n_ctx": 2048,
    "n_gpu_layers": -1, # Use -1 to offload all possible layers to GPU
    "verbose": True, # Set to False for less output in production
    # Add other LlamaCpp parameters as needed (e.g., temperature, top_p)
    "temperature": 0.7,
    "max_tokens": 512, # Limit response length
}

def load_llm() -> LlamaCpp:
    """
    Loads the LlamaCpp model instance compatible with Langchain.
    Uses the path from the environment variable MODEL_PATH.
    Initializes the model only once and returns the existing instance on subsequent calls.

    Returns:
        LlamaCpp: The loaded LlamaCpp model instance for Langchain use.

    Raises:
        ValueError: If the MODEL_PATH environment variable is not set or the file does not exist.
    """
    global llm_instance
    if llm_instance is None:
        model_path = os.getenv("MODEL_PATH")
        if not model_path:
            raise ValueError("MODEL_PATH environment variable not set.")
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found at path: {model_path}")

        print(f"Loading model from: {model_path} with params: {DEFAULT_LLAMA_PARAMS}...")
        try:
            llm_instance = LlamaCpp(
                model_path=model_path,
                **DEFAULT_LLAMA_PARAMS # Unpack parameters
            )
            print("Langchain LlamaCpp model loaded successfully.")
        except Exception as e:
            print(f"Error loading LlamaCpp model: {e}")
            raise # Re-raise the exception after logging
    return llm_instance

if __name__ == '__main__':
    # Example usage: Load the LlamaCpp model and test invocation
    try:
        llm = load_llm()
        print(f"Model Path: {llm.model_path}")
        print("LLM Loader (LlamaCpp) is working.")

        # Test loading again to check singleton pattern
        llm_again = load_llm()
        print(f"Is it the same instance? {llm is llm_again}")

        # Example invocation (requires model to be downloaded and path set correctly)
        # print("\nTesting model invocation...")
        # try:
        #     response = llm.invoke("Explain the concept of ReWOO in one sentence.")
        #     print(f"Model Response:\n{response}")
        # except Exception as invoke_error:
        #     print(f"Error during model invocation: {invoke_error}")

    except ValueError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}") 