# MCP Agent

![iamge](https://github.com/user-attachments/assets/e6d4ccc9-751e-44cf-bd3b-e2b4bd80e500)

## Introduction

MCP Agent is an AI assistant that runs on your computer. This AI assistant understands natural language commands and can perform various tasks using different tools. Through connection with external programs, it provides various functions such as file system access, web search, terminal command execution, and more.

## Key Features

* **Natural Language Understanding**: The AI understands and executes commands given in everyday language.
* **Diverse Task Execution**: Capable of file management, answering simple questions, information retrieval, and more.
* **Extensible Tools**: New features can be easily added through the MCP (Model Context Protocol).

## Running with Docker

### Prerequisites

![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white) 

### Installation and Setup

1. **Build Docker Image:**
   ```bash
   docker build -t mcp-agent .
   ```

2. **Download AI Model:**
   * This project uses `QwQ` models.
   * The model needs to be downloaded manually (it's a large file).
   * You can download it from Hugging Face Hub (search for QwQ models from the Qwen team).
   * Save the downloaded model file in the `./models` directory on your local system.

3. **Environment Configuration (`.env`):**
   * Create a `.env` file in the project root directory.
   * Sample `.env` file:

     ```
     MODEL_FILENAME=QwQ-LCoT-7B-Instruct-IQ4_NL.gguf
     MODEL_DIR=./models
     N_CTX=32768
     GPU_LAYERS=-1
     GRAMMAR_PATH=react_output.gbnf
     REACT_MAX_ITERATIONS=10
     LOG_LEVEL=INFO
     LOG_DIR=logs
     ```

4. **Tool Configuration (`mcp.json`):**
   * This file configures external tools for the AI assistant.
   * Create a `mcp.json` file in the project root directory.
   * Example configuration:

     ```json
     {
       "mcpServers": {
         "desktop-commander": {
           "command": "npx",
           "args": [
             "-y",
             "@mcp-commander/mcp-server-desktop-commander@latest",
             "listen"
            ]
         }
       }
     }
     ```

5. **Run Docker Container:**
   ```bash
   docker run -d --name mcp-agent -p 8000:8000 -v $(pwd)/models:/app/models -v $(pwd)/mcp.json:/app/mcp.json -v $(pwd)/.env:/app/.env -v $(pwd)/logs:/app/logs mcp-agent
   ```

## Using the AI Assistant

You can interact with the AI assistant through web browser or API calls.

**Example (using terminal):**

```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"text": "What is the weather today?"}'
```

Or access `http://localhost:8000` in your web browser to use the interactive interface.

## Adding New Tools

To expand the AI assistant's capabilities, you can add new MCP servers to the `mcp.json` file. The MCP protocol provides a standard way for AI to communicate with external tools.

## Recommended Models

We recommend using QwQ or Qwen series models for this project. These models have excellent reasoning capabilities and are optimized for tool use.

### Model Size Recommendations:
* **Recommended**: Models with 14B parameters or larger for optimal performance
* **Minimum**: 7B parameter models with 4-bit quantization
* **Not Recommended**: Models smaller than 7B parameters may not have sufficient reasoning capabilities for complex tasks

## License

This project is provided under the MIT License. For details, see the [LICENSE](LICENSE) file.
