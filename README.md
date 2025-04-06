# OS-Agnostic AI Agent with MCP Integration

This project implements an OS-agnostic AI agent that runs within a web container.
It utilizes an ONNX model for inference and integrates with external tools and data sources via the Model Context Protocol (MCP).

## Features

- **ONNX Model Inference:** Uses the `gemma-3-1b-it-ONNX` model (quantized) for text generation.
- **Runtime Model Download:** Downloads the ONNX model at runtime if not present locally, making the initial setup lightweight.
- **MCP Integration:** Communicates with MCP servers defined in `mcp.json` to leverage external tools (e.g., terminal interaction via `iterm-mcp`).
- **Web API:** Provides a FastAPI-based web interface for interacting with the agent.
- **Async Architecture:** Built using `asyncio` for efficient handling of I/O operations (model download, MCP communication).
- **E2E Tested:** Includes end-to-end tests to verify core functionalities like model inference and MCP tool calls.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Create a Python virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure MCP Servers (`mcp.json`):**
    Create a `mcp.json` file in the project root. Add the MCP servers you want the agent to use. Example using `iterm-mcp` (requires iTerm2 on macOS and Node.js/npx):
    ```json
    {
      "mcpServers": {
        "iterm-mcp": {
          "command": "npx",
          "args": [
            "-y",
            "@smithery/cli@latest",
            "run",
            "iterm-mcp",
            "--key",
            "YOUR_ITREM_MCP_KEY" 
          ]
        }
      }
    }
    ```
    *Replace* `YOUR_ITREM_MCP_KEY` *with your actual key if required by your iterm-mcp setup.*

## Running the Agent

```bash
# Ensure your virtual environment is activated
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- The agent will start, and if the model is not found in `./test_model_cache/`, it will begin downloading it (this may take some time).
- Configured MCP servers will also be started in the background.
- Access the API documentation at `http://localhost:8000/docs`.

## Key API Endpoints

- **`POST /v1/inference`**: Sends text to the AI model for generation.
  - **Body:** `{"text": "Your prompt here"}`
- **`GET /v1/mcp/{server_name}/status`**: Checks the status of a running MCP server.
- **`POST /v1/mcp/{server_name}/call`**: Calls a specific tool on an MCP server.
  - **Body:** `{"tool_name": "tool_to_call", "arguments": {"arg1": "value1"}}`

## Development & Testing

- **Running Tests:**
  ```bash
  # Run all tests
  pytest

  # Run only unit tests
  pytest -m unit

  # Run only E2E tests (requires model download and MCP setup)
  pytest -m e2e
  ```
- **Linting/Formatting:** (Configure tools like Flake8, Black, isort as needed)

## Project Structure

```
.gitignore
mcp.json          # User-configured MCP servers
README.md
requirements.txt
app/
├── api/          # FastAPI routers and endpoints
│   └── v1/
├── core/         # Configuration, settings
├── main.py       # FastAPI application entry point
├── models/       # Pydantic models (request/response)
├── mcp_client/   # Client for communicating with MCP servers
│   └── client.py
└── services/     # Business logic (inference, MCP management)
    ├── inference_service.py
    └── mcp_service.py
tests/
├── integration/  # Integration and E2E tests
│   └── test_e2e.py
└── unit/         # Unit tests
    └── ...
test_model_cache/ # (Ignored by Git) Downloaded ONNX model stored here
```

## Important Notes

- The agent currently performs single-step inference. It does not yet have complex reasoning or multi-step execution capabilities based on user prompts.
- The `iterm-mcp` server specifically requires macOS with iTerm2 installed. 