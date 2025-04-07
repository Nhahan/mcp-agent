# OS-Agnostic AI Agent with MCP Integration

[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](http://unlicense.org/)

This project provides a web-based AI agent designed to be OS-agnostic. It leverages the Model Context Protocol (MCP) to interact with external tools and data sources, enabling flexible and extensible AI capabilities.

## Project Structure

```
.
├── app/                  # Main application code
│   ├── api/              # API endpoint definitions (FastAPI)
│   ├── core/             # Core components (e.g., configuration)
│   ├── mcp_client/       # MCP client implementation
│   ├── prompts/          # Prompt templates for the LLM
│   ├── services/         # Business logic (Inference, MCP communication)
│   └── main.py           # FastAPI application entry point
├── models/               # Directory for downloaded GGUF model
├── logs/                 # Directory for application and ReAct logs
├── .env                  # Environment variables (API keys, paths)
├── requirements.txt      # Python dependencies
├── mcp.json              # MCP server configurations
├── Dockerfile            # (Optional) Dockerfile for containerization
├── README.md             # This file
├── GUIDE.md              # Detailed guide on MCP and development (Korean)
├── LICENSE               # Apache 2.0 License for the project code
├── LICENSE_GEMMA         # Gemma Model License
├── NOTICE_GEMMA          # Gemma Model Notice
└── TERMS_GEMMA           # Gemma Model Terms of Use
```

## Getting Started

### Prerequisites

*   Python 3.9+
*   `pip` (Python package installer)
*   (Optional) Docker and Docker Compose

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Set up environment variables:**
    *   Copy the example `.env.example` (if provided) to `.env`.
    *   Update `.env` with your specific configurations, especially:
        *   `MODEL_URL`: (Optional) URL to download the GGUF model if different from the default.
        *   `MODEL_PATH`: (Optional) Path where the model file should be stored locally (relative to project root). Defaults to `./models/Gemma-3-4B-Fin-QA-Reasoning.Q4_K_M.gguf`.
        *   `MCP_CONFIG_PATH`: (Optional) Path to the MCP configuration file. Defaults to `./mcp.json`.
        *   `LOG_LEVEL`: (Optional) Desired log level (e.g., `DEBUG`, `INFO`, `WARNING`). Defaults to `INFO`.
        *   `HUGGING_FACE_TOKEN`: (Optional) Your Hugging Face Hub token if needed for downloading private models or tokenizers.
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Agent

1.  **Configure MCP Servers:**
    *   Edit `mcp.json` to define the MCP servers you want the agent to use. See the example below. The agent will automatically attempt to start and connect to these servers.
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
            "<your-iterm-mcp-key>" // Replace with your actual key if needed
          ]
        },
        "sequential-thinking": {
          "command": "npx",
          "args": [
            "-y",
            "@modelcontextprotocol/server-sequential-thinking"
          ]
        }
      }
    }
    ```
    *   Ensure the specified commands (like `npx`) are available in your environment's PATH.

2.  **Start the application:**
    ```bash
    python -m app.main
    ```
    *   The first time you run it, the agent will download the **GGUF model (approx. 4.1GB+)**, which might take some time.
    *   The server will start, typically on `http://127.0.0.1:8000`.

### Using the API

Once the server is running, you can interact with the agent via its API endpoints:

*   **Health Check:** `GET /health`
    *   Provides the status of the agent, including model loading status and connected MCP servers.
*   **Chat Endpoint:** `POST /chat`
    *   Send user queries or commands in the request body as JSON:
        ```json
        {
          "text": "What files are in the current directory?"
        }
        ```
    *   The agent will process the request, potentially using MCP tools via the ReAct pattern, and return a JSON response:
        ```json
        {
          "response": "The command `ls -al` shows...",
          "thoughts_and_actions": [
            {
              "thought": "I need to list the files. I'll use the iterm-mcp/write_to_terminal tool.",
              "action": { "raw": "iterm-mcp/write_to_terminal(...)" },
              "observation": "Okay, the command was sent..."
            },
            {
              "thought": "Now I need to read the output.",
              "action": { "raw": "iterm-mcp/read_terminal_output(...)" },
              "observation": "total 4\n-rw-r--r-- 1 user user 0 Oct 28 10:00 file1.txt\n..."
            }
          ],
          "full_response": "User: What files are in the current directory?\nAction: ...\nObservation: ...\nAction: ...\nObservation: ...\n",
          "error": null,
          "log_session_id": "174406...",
          "log_path": "logs/react_logs/174406..."
        }
        ```

## Model Context Protocol (MCP)

MCP standardizes communication between Large Language Models (LLMs) and external tools/data sources. This agent uses MCP to:

*   Discover available tools from connected MCP servers.
*   Invoke tools with appropriate arguments based on the LLM's reasoning (ReAct pattern).
*   Receive observations (results) from tool executions.

To add or modify capabilities, simply update the `mcp.json` file with the desired server configurations. The agent dynamically loads and interacts with these servers. Refer to [GUIDE.md](GUIDE.md) for a more detailed explanation of MCP (in Korean).

## Logging

*   **Application Logs:** Standard application logs are output to the console. The level can be controlled via the `LOG_LEVEL` environment variable.
*   **ReAct Logs:** Detailed step-by-step logs for each `/chat` request (including thoughts, actions, observations, and LLM prompts/responses) are saved in the `logs/react_logs/<session_id>/` directory. This is invaluable for debugging the agent's reasoning process.

## License

This project's code is released into the public domain under The Unlicense. See the [LICENSE](LICENSE) file for details.

The Gemma model used in this project is subject to its own license and terms. Please review the following files carefully:

*   **Gemma License:** [LICENSE_GEMMA](LICENSE_GEMMA)
*   **Gemma Notice:** [NOTICE_GEMMA](NOTICE_GEMMA)
*   **Gemma Terms of Use:** [TERMS_GEMMA](TERMS_GEMMA)

By using this project, you agree to abide by both the project's license and the Gemma model's license and terms.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
