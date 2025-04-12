# MCP Agent Development Guide

## Project Overview

This project implements an AI agent that can utilize [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol/spec) servers to extend its capabilities. The AI agent is designed to be:

1. **Web-based**: Runs in a web container
2. **Lightweight**: Uses the ONNX runtime with a quantized model
3. **Extensible**: Can leverage any MCP server defined in `mcp.json`
4. **Language-agnostic**: Primarily operates in English internally, but can respond in the user's language

## Core Components

- **InferenceService**: Handles model loading, inference, and the ReAct pattern implementation
- **MCPService**: Manages MCP server connections and tool execution
- **API Layer**: FastAPI endpoints that handle requests and responses

## Key Features

### Language Handling and Translation Flow

1. **Input Processing**:
   - Detect the language of user input 
   - If Korean (ko), translate to English before sending to the AI model
   - For other languages, pass through as-is

2. **Internal Processing**:
   - AI operates entirely in English, regardless of input language
   - All prompts, system instructions, and tool descriptions are in English
   - ReAct pattern reasoning is performed in English

3. **Output Processing**:
   - If original input was Korean, translate the AI's response back to Korean
   - For other languages, return the AI's English response as-is

This approach ensures consistent reasoning quality while providing a natural user experience in different languages.

### MCP Server Integration

- The AI agent uses the MCP protocol to interact with external tools
- MCP servers are defined in `mcp.json` and loaded dynamically
- No hardcoded MCP servers - all tools must be discovered at runtime
- Tool schemas are cached to improve performance and reduce latency

### ReAct Pattern Implementation

The AI agent uses the ReAct (Reasoning + Acting) pattern:

1. **Thought**: The AI analyzes the query, available tools, and previous observations
2. **Action**: The AI either calls a tool or provides a final answer
3. **Observation**: Results from tool execution are provided back to the AI
4. **Repeat**: The process continues until a final answer is reached

### Logging System

- All interactions are logged to a centralized `meta.json` file per session
- Logs include user queries, AI responses, tool calls, and observations
- Logs are stored in `logs/api_logs/{session_id}/meta.json`

## Configuration Management (`app/core/config.py`)

All major configuration parameters are centralized in `app/core/config.py` using Pydantic's `BaseSettings`. This allows configuration via environment variables or a `.env` file.

**Key Configuration Variables:**

*   **Model Settings:**
    *   `MODEL_FILENAME`: Name of the model file (e.g., `QwQ-LCoT-7B-Instruct-IQ4_NL.gguf`).
    *   `MODEL_PATH`: Full path to the model file. If not set, calculated from `MODEL_DIR` and `MODEL_FILENAME`.
    *   `MODEL_DIR`: Directory where models are stored (defaults to `./models` or `/app/models` in Docker).
    *   `N_CTX`: Model context window size (default: 32768).
    *   `GPU_LAYERS`: Number of layers to offload to GPU (default: -1 for all possible).
    *   `GRAMMAR_PATH`: Path to the GBNF grammar file (default: `react_output.gbnf`).
*   **LLM Generation Parameters:**
    *   `MODEL_MAX_TOKENS`: Max tokens to generate (default: 1024).
    *   `MODEL_TEMPERATURE`: Sampling temperature (default: 0.7).
    *   `MODEL_TOP_P`: Top-P sampling nucleus (default: 0.9).
    *   `MODEL_TOP_K`: Top-K sampling limit (default: 40).
    *   `MODEL_MIN_P`: Min-P sampling (default: 0.05, model must support).
*   **ReAct Loop:**
    *   `REACT_MAX_ITERATIONS`: Maximum number of thought/action cycles (default: 10).
*   **MCP:**
    *   `MCP_CONFIG_FILENAME`: Name of the MCP configuration file (default: `mcp.json`).
    *   `MCP_CONFIG_PATH`: Full path to the MCP config file (calculated).
*   **Logging:**
    *   `LOG_LEVEL`: Logging level (e.g., INFO, DEBUG) (default: INFO).
    *   `LOG_DIR`: Directory for log files (default: `logs`).

**Usage in Code:**

Import the settings object and access attributes directly:

```python
from app.core.config import settings

# Example usage
max_iterations = settings.react_max_iterations
model_path = settings.model_path
```

## Development Rules

1. **No Hardcoding of MCP Servers**:
   - The system must detect and use MCP servers defined in `mcp.json`
   - The user can add or remove MCP servers at any time

2. **English-Only Internal Processing**:
   - All system prompts and internal reasoning must be in English
   - Translation must happen at the API boundary, not within the core logic

3. **Simplified Tool Descriptions**:
   - Initially provide only a list of available tools with brief descriptions
   - Provide detailed tool information only when a specific tool is selected

4. **Centralized Logging**:
   - All logs for a session must be stored in a single `meta.json` file
   - No file fragmentation or redundant logs

5. **Proper Error Handling**:
   - All errors must be properly captured and formatted for the user
   - Failed tool executions should not crash the system

6. **No Test Mocking**:
   - Tests must run against the actual system, not mocked components
   - Tests must not be skipped and should reflect real production scenarios

7. **Do Not Check Model Status in Translation**: 
   - Translation functions (e.g., `_translate_en_to_ko`) should **not** check the model's loading status. `ModelService` is responsible for managing its own state. Rely on `ModelService` to handle requests appropriately or raise errors if it's not ready.

8. **Variable Naming Consistency**:
   - Always ensure that variable names are consistent throughout the codebase
   - When a function returns a variable, make sure to use the same name when referring to it in the calling code
   - Pay special attention to function return values to avoid `NameError` exceptions

9. **Avoid Duplicate Definitions**:
   - Do not redefine functions or methods within other functions
   - Import utilities from the appropriate modules instead of redefining them
   - When modifying methods, scan the entire function for duplicate helper functions

10. **Regular Code Cleanup**:
   - Regularly scan the codebase for unused imports, variables, and functions
   - Remove or comment out unused code to improve readability and performance
   - Use linters and type checkers to identify potential issues before they cause runtime errors

11. **Proper Dependency Injection**:
   - Never use FastAPI's `Depends` in service class constructors directly
   - Service classes should import settings directly from `app.core.config`
   - The application uses singleton patterns for services, with instances created in `lifespan`
   - Use dependency injection only in API route functions or middleware

12. **Robust Logging Implementation**:
   - Always use defensive programming techniques when handling data of uncertain structure
   - Use duck typing and attribute/key access within try-except blocks instead of explicit type checking
   - Design functions to accept broader types using protocol/interface patterns
   - Implement graceful fallback strategies for unexpected data structures
   - When dealing with serialization, ensure proper error handling for non-serializable types

13. **Service Interface Contracts**:
   - When creating methods that other services depend on, document the method's contract clearly
   - Maintain backward compatibility when modifying existing service methods
   - Always implement all expected methods in services that other components depend on
   - Use descriptive error messages when a required method is missing
   - Check service interfaces during system startup rather than runtime where possible

## Model Requirements

- The project uses a quantized ONNX model: `gemma-3-1b-it-ONNX/model_q4f16.onnx`
- The model is ~1GB in size and should be downloaded at runtime, not hosted on GitHub
- The model should be loaded into memory only when needed

## Code Structure and Standards

- Code should be well-structured and maintainable
- Follow proper error handling and logging practices
- Avoid duplicate code and ensure good separation of concerns
- All user-facing messages should adapt to the user's language
- Internal system messages should be in English

## How to Test

- Run tests to ensure all components work correctly
- Check logs to verify proper flow of information
- Test with different languages to ensure translation works correctly
- Test with various MCP servers to ensure tool discovery and execution works

## Important Notes

- The agent must be able to run in a web container
- The agent must be OS-agnostic
- The agent must dynamically adapt to available MCP servers
- Always follow the ReAct pattern for reasoning and tool use 