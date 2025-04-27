# core/llm_loader.py
import os
from dotenv import load_dotenv
from llama_cpp import Llama
from typing import Optional

# Load environment variables from .env file
load_dotenv()

# Global variable to hold the loaded model instance
llm_instance: Optional[Llama] = None

def load_llm() -> Llama:
    """
    Loads the Llama model instance using the path from the environment variable.
    Initializes the model only once and returns the existing instance on subsequent calls.

    Returns:
        Llama: The loaded Llama model instance.

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

        # TODO: Add more configuration options as needed (e.g., n_ctx, n_gpu_layers)
        # These might need to be loaded from environment variables or a config file as well.
        print(f"Loading model from: {model_path}...")
        try:
            llm_instance = Llama(
                model_path=model_path,
                n_ctx=2048, # Example context size, adjust as needed
                n_gpu_layers=-1, # Use -1 to offload all possible layers to GPU
                verbose=True # Set to False for less output
            )
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise # Re-raise the exception after logging
    return llm_instance

if __name__ == '__main__':
    # Example usage: Load the model and print basic info
    try:
        llm = load_llm()
        print(f"Model Name: {os.path.basename(llm.model_path)}")
        # You can add more checks or example inference here if needed
        print("LLM Loader is working.")

        # Test loading again to check singleton pattern
        llm_again = load_llm()
        print(f"Is it the same instance? {llm is llm_again}")

    except ValueError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}") 