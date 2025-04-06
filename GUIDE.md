# AI Agent Development Guide

This guide provides an overview of the AI agent's architecture, focusing on how it leverages the Model Context Protocol (MCP) to interact with external tools and data.

## 1. Core Concept: AI Agent with Tools

The primary goal of this agent is to understand user instructions in natural language and fulfill them by combining its internal knowledge (from the LLM) with capabilities provided by external tools through MCP.

**Key Idea:** The user interacts *only* with the agent (via the `/v1/inference` API). The agent, upon receiving a request, *decides* whether to:
    a) Respond directly using its language model.
    b) Utilize an external tool via an MCP server to gather information or perform an action, and then use that result to formulate the final response.

## 2. Role of Model Context Protocol (MCP)

MCP acts as a standardized communication layer between the AI agent and external "tools" (MCP Servers).

- **MCP is an Agent's Tool, Not the User's:** Users do not directly interact with MCP servers or their tools. The agent internally manages connections to MCP servers defined in `mcp.json`.
- **Enabling Capabilities:** MCP allows the agent to perform tasks beyond simple text generation, such as:
    - Running commands in a terminal (e.g., using `iterm-mcp`).
    - Accessing real-time data (e.g., weather, stock prices via custom MCP servers).
    - Interacting with databases or APIs.

## 3. Agent Workflow Example (with MCP)

Let's trace a user request that involves an MCP tool:

1.  **User Request:** The user sends a POST request to `/v1/inference` with the prompt: `"터미널에서 pwd 실행해줘"` (Run `pwd` in the terminal).
2.  **Agent Analysis (Inference Service):**
    - The `InferenceService` receives the prompt.
    - It analyzes the prompt using predefined patterns (or potentially more advanced LLM-based reasoning in the future).
    - It detects the intent to run a terminal command (`pwd`).
3.  **MCP Tool Invocation:**
    - The `InferenceService` determines that the `iterm-mcp` server (if configured and running) is suitable for this task.
    - It calls the `MCPService`'s `call_mcp_tool` function, requesting to execute the `write_to_terminal` tool on `iterm-mcp` with the argument `{"command": "pwd"}`.
    - It then calls `call_mcp_tool` again to execute the `read_terminal_output` tool to capture the result.
4.  **Response Formulation:**
    - The `MCPService` returns the output received from `iterm-mcp` (e.g., `/Users/sunningkim/Developer/mcp-agent`).
    - The `InferenceService` formats this result into a user-friendly response, perhaps prefixing it with "Terminal Output:".
5.  **User Response:** The agent sends back a JSON response containing the formatted terminal output.

## 4. Implementation Details

- **`InferenceService` (`app/services/inference_service.py`):** Contains the core agent logic. It analyzes prompts and orchestrates calls to the LLM or MCP tools.
- **`MCPService` (`app/services/mcp_service.py`):** Manages the lifecycle of MCP server processes and handles the low-level JSON-RPC communication via `MCPClient`.
- **`MCPClient` (`app/mcp_client/client.py`):** Implements the actual JSON-RPC 2.0 communication over stdio with a single MCP server process.
- **Configuration (`mcp.json`):** Users define available MCP servers and how to run them in this file.

## 5. Current Limitations & Future Work

- **Simple Agent Logic:** The current agent logic relies on basic regular expressions to detect tool usage intent. More sophisticated natural language understanding and planning (e.g., using the LLM itself to decide which tool to use and how) are needed for complex tasks.
- **Limited Toolset:** The primary example uses `iterm-mcp`. Adding more diverse MCP servers would significantly expand the agent's capabilities.
- **Error Handling:** Robust error handling for MCP communication failures, tool execution errors, and unexpected outputs needs further development. 