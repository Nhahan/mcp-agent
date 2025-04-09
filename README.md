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

### Running with Docker

Using Docker provides a consistent environment across different operating systems.

1.  **Prepare `mcp.json`:**
    *   Create or modify an `mcp.json` file **on your host machine** (outside the container) with the desired MCP server configurations.

2.  **Build the Docker image:**
    From the project root directory (where the `Dockerfile` is located), run:
    ```bash
    docker build -t mcp-agent .
    ```
    This might take some time, especially the first time, as it downloads the base image and installs dependencies.

3.  **Run the Docker container:**
    ```bash
    docker run --rm -it -p 8000:8000 \
      -v $(pwd)/mcp.json:/app/mcp.json:ro \
      -v $(pwd)/logs:/app/logs \
      -v model_data:/app/models \
      mcp-agent
    ```
    *   `--rm`: Removes the container when it stops.
    *   `-it`: Runs in interactive mode with a pseudo-TTY.
    *   `-p 8000:8000`: Maps port 8000 on your host to port 8000 in the container.
    *   `-v $(pwd)/mcp.json:/app/mcp.json:ro`: Mounts your local `mcp.json` file into the container (read-only).
    *   `-v $(pwd)/logs:/app/logs`: Mounts a local `logs` directory into the container to persist logs.
    *   `-v model_data:/app/models`: Uses a Docker named volume (`model_data`) to store the downloaded model persistently across container runs. This prevents re-downloading the large model file every time.
    *   `mcp-agent`: The name of the image you built.

    *   The first time you run the container, it will download the model into the `model_data` volume.
    *   The agent will be accessible at `http://localhost:8000` on your host machine.

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

## License

This project's code is released into the public domain under The Unlicense. See the [LICENSE](LICENSE) file for details.

The Gemma model used in this project is subject to its own license and terms. Please review the following files carefully:

*   **Gemma License:** [LICENSE_GEMMA](LICENSE_GEMMA)
*   **Gemma Notice:** [NOTICE_GEMMA](NOTICE_GEMMA)
*   **Gemma Terms of Use:** [TERMS_GEMMA](TERMS_GEMMA)

By using this project, you agree to abide by both the project's license and the Gemma model's license and terms.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
