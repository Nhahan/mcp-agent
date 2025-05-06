from langchain_community.cache import InMemoryCache
import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, Any, AsyncGenerator
from langchain_core.globals import set_llm_cache
import argparse
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn

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

# Pydantic model for chat requests
class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None # Placeholder for future stateful conversations

class ChatResponse(BaseModel):
    response: Any
    conversation_id: str

# Lifespan manager for FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("API Startup: Lifespan event triggered.")

    # 1. Load LLM
    try:
        logger.info("API Startup: Loading LLM...")
        llm = load_llm()
        app.state.llm = llm
        logger.info("API Startup: LLM loaded successfully.")
    except Exception as e:
        logger.error(f"API Startup Error: Failed to load LLM: {e}", exc_info=True)
        raise RuntimeError("Failed to load LLM during startup.") from e

    # 2. Initialize MCP Client
    logger.info("API Startup: Loading MCP server configurations...")
    mcp_servers_config_dict = load_mcp_servers_from_config()
    if not mcp_servers_config_dict:
        logger.warning("API Startup: No enabled and valid MCP server configurations found. MCP Client will have no tools.")
        mcp_servers_config_dict = {}

    # The MultiServerMCPClient will be initialized and its servers started
    # when entering the 'async with' block.
    try:
        logger.info(f"API Startup: Initializing MultiServerMCPClient context...")
        mcp_client_instance = MultiServerMCPClient(mcp_servers_config_dict)
        app.state.mcp_client_manager = mcp_client_instance # Store the manager instance
        
        # Enter the context, which starts the servers
        mcp_client = await app.state.mcp_client_manager.__aenter__()
        app.state.mcp_client = mcp_client # Store the active client

        available_tools_raw = mcp_client.get_tools()
        tool_names = [tool.name for tool in available_tools_raw]
        logger.info(f"API Startup: MCP Client active. Discovered tools: {tool_names}")
    except Exception as e:
        logger.error(f"API Startup Error: Failed to initialize or start MCP Client: {e}", exc_info=True)
        # Ensure aexit is called if aenter succeeded partially or if an error occurs after aenter
        if hasattr(app.state, 'mcp_client_manager') and app.state.mcp_client_manager:
            try:
                await app.state.mcp_client_manager.__aexit__(type(e), e, e.__traceback__)
            except Exception as ae:
                logger.error(f"API Startup Error: Error during MCP client manager cleanup: {ae}", exc_info=True)
        raise RuntimeError("Failed to initialize or start MCP Client during startup.") from e

    # 3. Build the agent graph
    try:
        logger.info("API Startup: Building agent graph...")
        compiled_app_graph, base_graph_config = build_rewoo_graph(app.state.llm, app.state.mcp_client)
        app.state.compiled_app_graph = compiled_app_graph
        app.state.base_graph_config = base_graph_config
        logger.info("API Startup: Agent graph built successfully.")
    except Exception as e:
        logger.error(f"API Startup Error: Failed to build agent graph: {e}", exc_info=True)
        if hasattr(app.state, 'mcp_client_manager') and app.state.mcp_client_manager:
             await app.state.mcp_client_manager.__aexit__(type(e), e, e.__traceback__)
        raise RuntimeError("Failed to build agent graph during startup.") from e

    logger.info("API Startup complete. Application is ready.")
    yield  # Application is now running

    # Shutdown phase
    logger.info("API Shutdown: Lifespan event triggered for shutdown.")
    if hasattr(app.state, 'mcp_client_manager') and app.state.mcp_client_manager:
        try:
            logger.info("API Shutdown: Closing MCP Client...")
            # __aexit__ handles server shutdown
            await app.state.mcp_client_manager.__aexit__(None, None, None)
            logger.info("API Shutdown: MCP Client closed successfully.")
        except Exception as e:
            logger.error(f"API Shutdown Error: Error closing MCP Client: {e}", exc_info=True)
    logger.info("API Shutdown complete.")


app = FastAPI(lifespan=lifespan)

@app.post("/chat", response_model=ChatResponse)
async def run_agent_query(request: Request, chat_request: ChatRequest):
    logger.info(f"Received chat request: {chat_request.message[:100]}...") # Log first 100 chars
    
    compiled_app = request.app.state.compiled_app_graph
    base_graph_config = request.app.state.base_graph_config

    query = chat_request.message
    initial_state = {
        "original_query": query,
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
    # config = {"configurable": base_graph_config} # Original line

    # Create the config dictionary for astream_events.
    # recursion_limit should be a top-level key in the config dict passed to stream/invoke methods.
    # base_graph_config (which holds "configurable") is for node-level configurations.
    
    # Start with a copy of base_graph_config if it might contain other top-level Pregel options
    # For this specific case, we know base_graph_config is primarily for "configurable" node inputs.
    config_for_stream = {
        "recursion_limit": 30, # Increase limit
        "configurable": base_graph_config.get("configurable", {}).copy() # Keep existing configurable items
    }
    
    final_answer_collected = None
    full_event_log = []

    try:
        logger.info(f"Invoking agent with query: '{query}' and effective recursion_limit: {config_for_stream.get('recursion_limit')}")
        async for event in compiled_app.astream_events(initial_state, config=config_for_stream, version="v1"):
            full_event_log.append(event) # For debugging
            kind = event['event']
            name = event.get('name', '')
            # logger.debug(f"Event: {kind}, Name: {name}, Data: {event['data']}")

            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    # Process streamed content if needed, e.g., for real-time response
                    pass
            elif kind == "on_tool_start":
                logger.info(f"Tool Started: {event['name']} with input {event['data'].get('input')}")
            elif kind == "on_tool_end":
                logger.info(f"Tool Ended: {event['name']}")
                # logger.info(f"Tool Output: {event['data'].get('output')}")

            # Check for the final answer or if the graph run has ended
            # The structure of the final event might vary depending on how the graph is defined.
            # Assuming 'generate_final_answer' node output is the final result.
            if kind == "on_chain_end" and name == "generate_final_answer": # Check if it's the end of the specific node
                 output = event['data'].get('output')
                 if output and isinstance(output, dict) and "final_answer" in output:
                    final_answer_collected = output["final_answer"]
                    logger.info(f"Final answer collected from 'generate_final_answer' node: {final_answer_collected}")
                    break # Exit loop once final answer is explicitly collected

        # If not collected by specific node check, try to get it from the last state of the graph stream
        if final_answer_collected is None and full_event_log:
            last_event = full_event_log[-1]
            if last_event['event'] == 'on_graph_end' or last_event['event'] == 'on_chain_end': # Broader check
                output_data = last_event['data'].get('output', {})
                if isinstance(output_data, dict):
                    final_answer_collected = output_data.get('final_answer') # Attempt to get from ReWOOState structure
                    if final_answer_collected:
                         logger.info(f"Final answer collected from graph end event: {final_answer_collected}")


        if final_answer_collected is None:
            logger.warning("Final answer was not explicitly collected. This might indicate an issue or an unexpected graph flow.")
            # Fallback: could inspect the last known state from events if necessary
            final_answer_collected = "Agent finished processing, but no explicit final answer was captured."


        # conversation_id for now is just a pass-through if provided, or a new one could be generated
        # For stateless ReWOO per call, conversation_id might not be strictly necessary for context yet.
        current_conversation_id = chat_request.conversation_id or "conv_" + asyncio.Lock()._get_loop().time().hex()

        return ChatResponse(response=final_answer_collected, conversation_id=current_conversation_id)

    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def visualize_graph_main():
    """Separate main function for visualization to avoid FastAPI app conflicts."""
    logger.info("Starting ReWOO Agent for visualization...")
    try:
        llm = load_llm()
    except Exception as e:
        logger.error(f"Failed to load LLM for visualization: {e}", exc_info=True)
        return

    mcp_servers_config_dict = load_mcp_servers_from_config()
    if not mcp_servers_config_dict:
        mcp_servers_config_dict = {}
    
    try:
        async with MultiServerMCPClient(mcp_servers_config_dict) as mcp_client:
            logger.info("Building agent graph for visualization...")
            # Pass mcp_client directly as it's now the active client from the context manager
            compiled_app_graph, _ = build_rewoo_graph(llm, mcp_client, visualize=True)
            logger.info("Graph visualization generated as graph_visualization.png.")
    except ImportError as e:
        logger.error(f"Visualization failed: {e}. Make sure pygraphviz and graphviz are installed.")
    except Exception as e:
        logger.error(f"Failed during visualization: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ReWOO Agent FastAPI server or visualize the graph.")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate a visualization of the LangGraph graph and save it as graph_visualization.png, then exit."
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for the FastAPI server.")
    parser.add_argument("--port", type=int, default=8000, help="Port for the FastAPI server.")

    args = parser.parse_args()

    if args.visualize:
        asyncio.run(visualize_graph_main())
    else:
        logger.info(f"Starting FastAPI server on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
