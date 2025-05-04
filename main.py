import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
import uuid # For generating unique conversation IDs
import argparse

# --- FastAPI Imports ---
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
# --- End FastAPI Imports ---

from core.llm_loader import load_llm
from agent.graph import build_rewoo_graph
from langchain_mcp_adapters.client import MultiServerMCPClient

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)

# Set the global LLM cache to in-memory
logger.info("Setting up global LLM cache (InMemory)...")
set_llm_cache(InMemoryCache())
logger.info("LLM Cache setup complete.")

# Simplified function to load server configs directly as per user mcp.json
def load_mcp_servers_from_config(config_path: Path = Path("mcp.json")) -> Dict[str, Dict[str, Any]]:
    """Loads server configurations from the JSON file and infers stdio transport if needed."""
    if not config_path.exists():
        logger.warning(f"MCP config file not found at {config_path}. Returning empty dict.")
        return {}
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        servers_config = config_data.get("servers", {})

        # Infer transport type if missing
        for server_name, config in servers_config.items():
            if "transport" not in config:
                if "command" in config and "args" in config:
                    config["transport"] = "stdio"
                    logger.info(f"Inferred 'stdio' transport for server '{server_name}'.")
                elif "url" in config:
                    logger.warning(f"Transport not specified for server '{server_name}' with URL. Relying on MultiServerMCPClient or later connection logic.")
                else:
                     logger.warning(f"Cannot infer transport type for server '{server_name}'. Skipping or check config.")
                     # Optionally remove this config or raise an error

        logger.info(f"Loaded and potentially adjusted server configurations for {len(servers_config)} server(s) from {config_path}.")
        return servers_config

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding MCP config file {config_path}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error loading MCP config {config_path}: {e}", exc_info=True)
        return {}

# --- FastAPI App Setup ---
app = FastAPI(
    title="MCP Agent API",
    description="API for interacting with the ReWOO LangGraph agent.",
    version="0.1.0"
)

# --- Global Variables for Agent and Client (Load once) ---
llm = None
mcp_client = None
compiled_agent_graph = None
agent_config = None
conversation_histories: Dict[str, List[Dict[str, Any]]] = {} # Store conversation history in memory

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    error: Optional[str] = None

# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    """Load LLM, MCP Client, and build graph on startup."""
    global llm, mcp_client, compiled_agent_graph, agent_config

    logger.info("API Startup: Loading LLM...")
    try:
        llm = load_llm()
        logger.info("API Startup: LLM loaded successfully.")
    except Exception as e:
        logger.error(f"API Startup Error: Failed to load LLM: {e}", exc_info=True)
        raise RuntimeError("Failed to load LLM during startup.") from e

    logger.info("API Startup: Loading MCP server configurations...")
    mcp_servers_config_dict = load_mcp_servers_from_config()
    if not mcp_servers_config_dict:
        logger.warning("API Startup: No enabled MCP servers found. Client will have no tools.")
        mcp_servers_config_dict = {}

    logger.info("API Startup: Initializing MultiServerMCPClient...")
    try:
        # Note: We manage the client lifecycle manually here instead of using 'async with'
        # because it needs to persist across requests. We'll close it on shutdown.
        mcp_client = MultiServerMCPClient(mcp_servers_config_dict)
        await mcp_client.start_all_servers() # Start servers manually
        available_tools_raw = mcp_client.get_tools()
        tool_names = [tool.name for tool in available_tools_raw]
        logger.info(f"API Startup: MCP Client initialized. Tools: {tool_names}")
    except Exception as e:
        logger.error(f"API Startup Error: Failed to initialize MCP Client: {e}", exc_info=True)
        raise RuntimeError("Failed to initialize MCP Client during startup.") from e

    logger.info("API Startup: Building agent graph...")
    try:
        compiled_agent_graph, base_graph_config = build_rewoo_graph(llm, mcp_client)
        agent_config = {"configurable": base_graph_config} # Store the base config
        logger.info(f"API Startup: Agent graph built. Config keys: {list(agent_config['configurable'].keys())}")
    except Exception as e:
        logger.error(f"API Startup Error: Failed to build agent graph: {e}", exc_info=True)
        if mcp_client:
            await mcp_client.shutdown() # Ensure client is shut down if graph fails
        raise RuntimeError("Failed to build agent graph during startup.") from e

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown MCP client gracefully."""
    global mcp_client
    logger.info("API Shutdown: Shutting down MCP Client...")
    if mcp_client:
        await mcp_client.shutdown()
        logger.info("API Shutdown: MCP Client shutdown complete.")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Endpoint to handle chat interactions with the agent."""
    global compiled_agent_graph, agent_config, conversation_histories

    if not compiled_agent_graph or not agent_config:
        raise HTTPException(status_code=503, detail="Agent is not ready. Please try again later.")

    conversation_id = request.conversation_id
    user_message = request.message

    # Manage conversation history
    if conversation_id and conversation_id in conversation_histories:
        history = conversation_histories[conversation_id]
        logger.info(f"Continuing conversation: {conversation_id}")
    else:
        conversation_id = str(uuid.uuid4())
        history = []
        conversation_histories[conversation_id] = history
        logger.info(f"Starting new conversation: {conversation_id}")

    # Append user message to history (LangChain expects specific format, adjust if needed)
    # Assuming ReWOOState uses a 'messages' list like [{role: 'user', content: '...'}, {role: 'assistant', content: '...'}]
    # For ReWOO, we might just pass the latest query? Let's adapt the state based on ReWOOState.
    # ReWOOState expects 'original_query'. We'll use the latest user message as the query.
    # If we need multi-turn ReWOO, the state/prompts might need adjustment.
    # For now, treat each call as a new ReWOO execution with the given message.

    initial_state = {
        "original_query": user_message,
        "plan": [],
        "current_step_index": 0,
        "current_tool_call": None,
        "prepared_tool_input": None,
        "evidence": {},
        "final_answer": None,
        "error_message": None,
        "max_retries": 2,
        "current_retry": 0,
        "workflow_status": None,
        "next_node": None
    }
    logger.info(f"Executing agent for conversation {conversation_id} with query: '{user_message}'")

    final_state = None
    error_message = None
    try:
        # Stream the results and collect the final state
        async for event in compiled_agent_graph.astream_events(
            initial_state,
            config=agent_config,
            version="v1",
        ):
            # You can add logic here to handle intermediate events if needed
            # For example, logging tool calls, etc.
             kind = event['event']
             if kind == "on_chat_model_stream":
                 content = event["data"]["chunk"].content
                 # if content: print(content, end="|") # Optional: stream intermediate LLM tokens
             elif kind == "on_tool_start":
                 logger.debug(f"Tool Start: {event['name']} Input: {event['data'].get('input')}")
             elif kind == "on_tool_end":
                 logger.debug(f"Tool End: {event['name']} Output: {event['data'].get('output')}")

            # The final state is typically available when the stream ends
            # Check if the event contains the final state (might need specific handling based on LangGraph version/setup)
            # A simple approach: Get the state from the last event or after the loop.
            # For now, let's get the final output after the loop finishes.

        # After the stream finishes, get the final state
        final_state_result = await compiled_agent_graph.ainvoke(initial_state, config=agent_config)
        final_answer = final_state_result.get("final_answer")
        error_message = final_state_result.get("error_message")

        if error_message:
             logger.error(f"Agent execution failed for conversation {conversation_id}: {error_message}")
             return ChatResponse(response="", conversation_id=conversation_id, error=f"Agent error: {error_message}")
        elif final_answer:
             logger.info(f"Agent execution successful for conversation {conversation_id}. Response generated.")
             # Optional: Append assistant response to history if maintaining multi-turn state
             # conversation_histories[conversation_id].append({"role": "assistant", "content": final_answer})
             return ChatResponse(response=final_answer, conversation_id=conversation_id)
        else:
             logger.error(f"Agent execution finished for conversation {conversation_id} but no final answer was generated.")
             return ChatResponse(response="", conversation_id=conversation_id, error="Agent finished without generating a final answer.")

    except Exception as e:
        logger.error(f"Error during agent execution for conversation {conversation_id}: {e}", exc_info=True)
        # Return an HTTP exception or a JSON response with error
        # raise HTTPException(status_code=500, detail=f"Agent execution failed: {e}")
        return ChatResponse(response="", conversation_id=conversation_id, error=f"Internal server error: {e}")


# --- Visualization Logic (Moved outside main execution flow) ---
async def generate_graph_visualization():
    """Generates and saves the graph visualization."""
    # Need LLM and MCP client temporarily just to build the graph structure
    temp_llm = None
    temp_mcp_client = None
    try:
        logger.info("Visualization: Loading LLM...")
        temp_llm = load_llm()
        logger.info("Visualization: LLM loaded.")

        logger.info("Visualization: Loading MCP config...")
        mcp_conf = load_mcp_servers_from_config()
        logger.info("Visualization: Initializing temporary MCP Client...")
        async with MultiServerMCPClient(mcp_conf) as temp_mcp_client:
            logger.info("Visualization: Building graph structure...")
            graph, _ = build_rewoo_graph(temp_llm, temp_mcp_client) # We only need the graph object

            logger.info("Attempting to generate graph visualization...")
            try:
                # Check if pygraphviz is available and draw
                from PIL import Image
                import io
                png_bytes = graph.get_graph().draw_png()
                img = Image.open(io.BytesIO(png_bytes))
                img.save("graph_visualization.png")
                logger.info("Graph visualization saved as graph_visualization.png")
            except ImportError:
                logger.error("Failed to generate visualization: `pygraphviz` not installed. Please install it (`pip install pygraphviz`).")
            except Exception as e:
                 # Catching graphviz execution errors specifically if possible
                if "failed to execute PosixPath" in str(e) or "Graphviz's executables not found" in str(e):
                     logger.error("Failed to generate visualization: Graphviz executables not found. Make sure Graphviz is installed and in your system's PATH (e.g., `brew install graphviz` or `sudo apt-get install graphviz`).")
                else:
                     logger.error(f"An unexpected error occurred during graph visualization: {e}", exc_info=True)

    except Exception as e:
        logger.error(f"Failed during visualization setup: {e}", exc_info=True)
    finally:
        # No explicit cleanup needed for temp client due to 'async with'
         logger.info("Visualization process finished.")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ReWOO Agent API or generate graph visualization.")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate a visualization of the LangGraph graph and save it as graph_visualization.png, then exit."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to run the API server on."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number to run the API server on."
    )
    args = parser.parse_args()

    if args.visualize:
        asyncio.run(generate_graph_visualization())
    else:
        # Run the FastAPI server
        logger.info(f"Starting FastAPI server on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
