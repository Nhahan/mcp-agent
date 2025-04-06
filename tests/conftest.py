import pytest
import logging
import os
from pathlib import Path
import datetime

# Define the root logger for configuration
logger = logging.getLogger()
app_logger = logging.getLogger("app") # Get the specific app logger

@pytest.fixture(scope="session", autouse=True)
def configure_test_logging(request):
    """Configure file logging for the entire test session."""
    log_dir = Path("logs/test")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a timestamped log file for each session
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"e2e_test_run_{timestamp}.log"
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG) # Log everything to file
    
    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    
    # Add handler to the root logger
    # Ensure we don't add it multiple times if tests are run in parallel or re-run
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file) for h in logger.handlers):
        logger.addHandler(file_handler)
        print(f"\nINFO: Test session logs will be saved to: {log_file}\n") # Print log file location

    # Set levels for noisy loggers if needed
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("onnxruntime").setLevel(logging.WARNING) # Reduce ONNX Runtime verbosity
    
    # Ensure root AND app logger levels allow DEBUG messages to pass to the handler
    original_root_level = logger.level
    original_app_level = app_logger.level
    
    # Set levels to DEBUG for the test session
    if original_root_level == logging.NOTSET or original_root_level > logging.DEBUG:
         logger.setLevel(logging.DEBUG)
    if original_app_level == logging.NOTSET or original_app_level > logging.DEBUG:
         app_logger.setLevel(logging.DEBUG)
         app_logger.propagate = True # Ensure messages go to root handlers

    yield # Run tests

    # Teardown: Remove handler and restore levels after session
    logger.removeHandler(file_handler)
    file_handler.close()
    
    # Restore original levels if they were changed
    if original_root_level != logger.level:
         logger.setLevel(original_root_level)
    if original_app_level != app_logger.level:
         app_logger.setLevel(original_app_level)
         # Restore propagation if needed, though default is usually True
         # app_logger.propagate = ... # Restore based on initial state if necessary 