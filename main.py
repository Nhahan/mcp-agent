import langchain
from langchain_community.cache import InMemoryCache
import asyncio
import logging # Add logging import
import json # Import json for parsing mcp.json
from pathlib import Path # Import Path for file handling
from typing import List, Dict, Any
from langchain_core.globals import set_llm_cache
# Import END from langgraph
from langgraph.graph import END

# Import necessary components
from core.llm_loader import load_llm
from agent.graph import build_rewoo_graph
from langchain_mcp_adapters.client import MultiServerMCPClient # Import MCP client

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
                    # Assuming SSE or Websocket based on URL presence, let MultiServerMCPClient handle specifics later if needed
                    # Or be more specific if possible, e.g., checking protocol in URL
                    logger.warning(f"Transport not specified for server '{server_name}' with URL. Relying on MultiServerMCPClient or later connection logic.")
                    # If necessary, could default to sse or websocket here, but might be risky
                    # config["transport"] = "sse" # Example default if needed
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

async def main():
    logger.info("Starting ReWOO Agent...")

    # 1. Load LLM
    try:
        logger.info("Loading LLM...")
        llm = load_llm()
        logger.info("LLM loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load LLM: {e}", exc_info=True)
        return

    # 2. Initialize MCP Client using mcp.json
    logger.info("Loading MCP server configurations from mcp.json...")
    mcp_servers_config_dict = load_mcp_servers_from_config()
    if not mcp_servers_config_dict:
        logger.warning("No enabled and valid MCP server configurations found in mcp.json. MCP Client will have no tools.")
        # Assign an empty list if no config is found, so the client can still initialize
        mcp_servers_config_dict = {}

    try:
        # Pass the dictionary directly to the client
        logger.info(f"Initializing MultiServerMCPClient with server configs: {mcp_servers_config_dict}")
        # Use async with for automatic resource management (including process termination)
        async with MultiServerMCPClient(mcp_servers_config_dict) as mcp_client:
            available_tools_raw = mcp_client.get_tools() # This should return List[BaseTool]
            # Log the names of discovered tools
            tool_names = [tool.name for tool in available_tools_raw]
            logger.info(f"MCP Client initialized. Discovered tools: {tool_names}")

            # --- [REMOVED TEMPORARY DIAGNOSTIC CODE for current_time tool] ---

            # 3. Build the agent graph/executor using the initialized tools
            try:
                logger.info("Building agent graph...")
                # Use the imported build_rewoo_graph function
                # It returns a tuple: (compiled_app, entry_node_name)
                # Unpack the tuple correctly
                compiled_app, entry_node_name = await build_rewoo_graph(llm, mcp_client) # Pass mcp_client now
                logger.info(f"Agent graph built successfully. Entry node: {entry_node_name}")
            except Exception as e:
                logger.error(f"Failed to build agent graph: {e}", exc_info=True)
                return # Exit if graph building fails

            # 4. Start interaction loop or process a single query
            # query = "What is the current time in Seoul?" # Example query
            query = "How do I bake a simple cake?" # Changed query for sequential-thinking

            logger.info(f"Starting agent execution with query: '{query}'")
            try:
                # Define the initial state according to ReWOOState TypedDict
                initial_state = {
                    "original_query": query,
                    "plan": None, # Plan is generated by the planner node
                    "current_step_index": 0,
                    "current_tool_call": None,
                    "prepared_tool_input": None,
                    "evidence": {}, # Initialize evidence as an empty dict
                    "final_answer": None,
                    "error_message": None,
                    "max_retries": 2, # Default max retries
                    "current_retry": 0,
                    "workflow_status": None, # Initial status
                    "next_node": None
                }
                logger.info(f"Initial state for agent: {initial_state}")

                # Format tool descriptions for the planner prompt
                tool_descriptions = "\n".join([
                    f"- {tool.name}: {tool.description}" # Use .name and .description attributes
                    for tool in available_tools_raw
                ]) if available_tools_raw else "No tools available."

                # Create the config dictionary required by the nodes
                config = {"configurable": {"llm": llm, "tools_str": tool_descriptions, "tools": available_tools_raw}}
                logger.info(f"Config for agent execution (tool names): {{key: val for key, val in config[\'configurable\'].items() if key != \'llm\'}} ...") # Avoid logging entire LLM object

                # Use the compiled_app object for streaming with the correct initial state and config
                async for event in compiled_app.astream_events(
                    initial_state, # Pass the complete initial state dictionary
                    config=config, # Pass the configuration dictionary
                    version="v1",
                    # types=["transform"], # Optional: filter event types
                    # tags=["some_tag"],  # Optional: filter by tags
                ):
                    kind = event['event']
                    if kind == "on_chat_model_stream":
                        content = event["data"]["chunk"].content
                        if content:
                            # Empty content in the context of OpenAI means no delta text
                            # print(content, end="|") # For direct streaming output
                            pass # We are logging steps, not streaming output directly for now
                    elif kind == "on_tool_start":
                        tool_name = event['name']
                        tool_input = event['data'].get('input')
                        logger.info(f"\n\nStarting tool: {tool_name} with inputs: {tool_input}")
                    elif kind == "on_tool_end":
                        tool_name = event['name']
                        tool_output = event['data'].get('output')
                        logger.info(f"Tool {tool_name} finished.")
                        logger.info(f"Tool output was: {tool_output}")
                        logger.info("-" * 40) # Separator after tool use
                    # Optional: Add more specific event handling if needed
                    # elif kind == "on_chain_start":
                    #     logger.debug(f"Chain started: {event['name']}")
                    # elif kind == "on_chain_end":
                    #     logger.debug(f"Chain finished: {event['name']}")
                    # elif kind == "on_chat_model_start":
                    #     logger.debug("Chat model started.")
                    # elif kind == "on_chat_model_end":
                    #     logger.debug("Chat model finished.")
                    # elif kind == "on_llm_start":
                    #     logger.debug("LLM started.")
                    # elif kind == "on_llm_end":
                    #     logger.debug("LLM finished.")
                    # elif kind == "on_llm_new_token":
                    #     logger.debug(f"New token: {event['data']}")
                    else:
                        # Catch-all for other event types for debugging
                        # logger.debug(f"Unhandled event type: {kind} | Data: {event['data']}")
                        pass # Keep log clean for now

                # After the stream, the full interaction is logged above.
                # If you need the final accumulated state, LangGraph typically provides it,
                # but the exact way depends on the graph's output node configuration.
                # For now, logging the stream events provides detailed insight.
                logger.info("Agent execution stream finished.")

            except Exception as e:
                logger.error(f"Error during agent execution: {e}", exc_info=True)

    except Exception as e:
        # Catch exceptions during MCP client initialization or within the async with block
        logger.error(f"Failed to initialize or use MultiServerMCPClient: {e}", exc_info=True)
        return # Exit if client fails

    logger.info("Agent main function finished.")

if __name__ == "__main__":
    asyncio.run(main())
