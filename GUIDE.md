# AI Agent Development Guide

This guide provides an overview of the AI agent's architecture, focusing on how it leverages the Model Context Protocol (MCP) to interact with external tools and data.

## 1. Core Concept: AI Agent with Tools

The primary goal of this agent is to understand user instructions in natural language and fulfill them by combining its internal knowledge (from the LLM) with capabilities provided by external tools through MCP.

**Key Idea:** The user interacts *only* with the agent (via the `/chat` API). The agent, upon receiving a request, *decides* whether to:
    a) Respond directly using its language model.
    b) Utilize an external tool via an MCP server to gather information or perform an action, and then use that result to formulate the final response.

## 2. Role of Model Context Protocol (MCP)

MCP is an open protocol that standardizes how AI applications provide context to large language models (LLMs). Similar to how USB-C provides a standardized way to connect devices to various peripherals, MCP provides a standardized way to connect AI models to different data sources and tools.

### MCP Architecture

MCP follows a client-server architecture:

- **MCP Hosts:** Programs like Claude Desktop, IDEs, or AI tools (including our agent) that want to access data through MCP
- **MCP Clients:** Protocol clients that maintain 1:1 connections with servers
- **MCP Servers:** Lightweight programs that expose specific capabilities through the standardized Model Context Protocol
- **Data Sources:** Local computer files, databases, services, or remote APIs that MCP servers can access

### Benefits of MCP

- **Standardization:** Provides a consistent way for AI models to interact with external tools and data
- **Interoperability:** Allows flexibility to switch between LLM providers and vendors
- **Security:** Implements best practices for securing data within your infrastructure
- **Extensibility:** Enables easy integration with a growing ecosystem of pre-built tools

### In Our Agent

- **MCP is an Agent's Tool, Not the User's:** Users do not directly interact with MCP servers or their tools. The agent internally manages connections to MCP servers defined in `mcp.json`.
- **Enabling Capabilities:** MCP allows the agent to perform tasks beyond simple text generation, such as:
    - Running commands in a terminal (e.g., using `iterm-mcp`).
    - Accessing real-time data (e.g., weather, stock prices via custom MCP servers).
    - Interacting with databases or APIs.

### MCP Code Example (Python)

Here's a practical example of how an AI agent interacts with an MCP server using the official MCP Python SDK:

```python
# --- MCP Server setup (for demonstration) ---
# (In practice, the server might be a separate process/file)
from mcp.server.fastmcp import FastMCP

# Create an MCP server instance with a name and default capabilities
mcp_server = FastMCP("DemoServer")

# Define a simple tool on the server: add(a, b) -> a + b
@mcp_server.tool()  # using the SDK's decorator to register a tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

# --- MCP Client (AI Agent side) ---
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_agent():
    # Configure the server process launch (using stdio transport)
    server_params = StdioServerParameters(
        command="python",
        args=["-c", "from server_script import mcp_server; mcp_server.run()"]  
        # ^ here we assume the server code above is saved in 'server_script.py'.
    )
    # Launch the MCP server as a subprocess and connect
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 1. Initialize handshake
            await session.initialize()  # exchanges capabilities
            # 2. (Optionally) discover available tools/resources
            tools = await session.list_tools()
            print("Available tools:", [t.name for t in tools])
            # 3. Invoke the "add" tool
            result = await session.call_tool("add", arguments={"a": 2, "b": 3})
            print("Tool result:", result)  # expected output: 5

# Run the async agent
asyncio.run(run_agent())
```

#### Communication Details

When `call_tool("add", ...)` is executed, the client sends a JSON-RPC message like:

```json
{ 
  "jsonrpc": "2.0", 
  "id": 1, 
  "method": "tool/execute", 
  "params": { 
    "name": "add", 
    "arguments": {"a": 2, "b": 3} 
  } 
}
```

The server then responds with:

```json
{ 
  "jsonrpc": "2.0", 
  "id": 1, 
  "result": { 
    "value": 5 
  } 
}
```

All of this JSON-RPC communication is abstracted by the SDK, allowing developers to focus on their application logic rather than protocol details.

## 3. Agent Workflow Example (with MCP)

Let's trace a user request that involves an MCP tool:

1.  **User Request:** The user sends a POST request to `/v1/chat` with the prompt: `"터미널에서 pwd 실행해줘"` (Run `pwd` in the terminal).
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
- **Expanding MCP Ecosystem:** Integrating with a wider range of MCP servers to provide access to various data sources and tools. 
  