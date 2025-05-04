from langchain_community.cache import InMemoryCache
import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, Any
from langchain_core.globals import set_llm_cache
import argparse

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

async def main(visualize: bool = False):
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
        logger.info(f"Initializing MultiServerMCPClient with server configs: {mcp_servers_config_dict}")
        # Use async with for automatic resource management (including process termination)
        async with MultiServerMCPClient(mcp_servers_config_dict) as mcp_client:
            available_tools_raw = mcp_client.get_tools()
            tool_names = [tool.name for tool in available_tools_raw]
            logger.info(f"MCP Client initialized. Discovered tools: {tool_names}")

            # 3. Build the agent graph/executor using the initialized tools
            try:
                logger.info("Building agent graph...")
                compiled_app, base_graph_config = build_rewoo_graph(llm, mcp_client)

            except Exception as e:
                logger.error(f"Failed to build agent graph: {e}", exc_info=True)
                return # Exit if graph building fails

            # 4. Start interaction loop or process a single query
            query = """Generate a highly detailed and technically accurate recipe for a **'Strawberry Swirl Chocolate Milk Cake'**, aiming for the quality expected from an expert baker, according to the following stringent requirements:

**Content Requirements:**
* The recipe must describe a moist chocolate cake batter explicitly using **chocolate milk** as a primary liquid ingredient (instead of regular milk). Ensure this ingredient is clearly listed and used in the instructions.
* It must detail the preparation of a **sweetened fresh strawberry puree** and the technique for **swirling** it into the batter before baking to create a visible marbled effect.
* It is crucial that the instructions for the **cake batter** and the **strawberry puree swirl** are presented in two **completely separate sections**, each under its own specific H2 heading ('## Cake Instructions' and '## Puree Instructions') and containing its own numbered steps.
* All measurements must be provided in **US customary units** (cups, tsp, tbsp, °F).
* The recipe must conclude with a **'Notes'** section containing at least two **highly specific and practical** baking tips directly relevant to potential challenges or enhancements for this particular 'Strawberry Swirl Chocolate Milk Cake' (e.g., ensuring swirl visibility is maintained during baking, how the properties of chocolate milk might affect the batter, tips for achieving maximum moistness).

**Formatting Requirements:**
* Pay close attention to the following Markdown formatting rules, ensuring the output strictly adheres to them for the document intended for `/Users/sunningkim/Developer/recipe.md`.
* The main title must be a level 1 heading (`#`).
* Section headings ('Ingredients', 'Cake Instructions', 'Puree Instructions', 'Notes') must be level 2 headings (`##`).
* The list of ingredients must use bullet points (`-`). Within each bullet point, the primary ingredient name *must* be formatted in **bold** text, followed by a colon, and then the quantity (Example: `- **All-purpose flour**: 2 cups`). Ensure this format is followed exactly for all ingredients.
* All procedural steps within the 'Cake Instructions' and 'Puree Instructions' sections must be presented as numbered lists (`1.`).

**Final Action Sequence:**
After generating the content that meticulously follows all content and formatting requirements, hypothetically write this precise Markdown content to the file path `/Users/sunningkim/Developer/recipe.md`, confirm successful creation, and then read the exactly formatted Markdown content back to me."""

            logger.info(f"Starting agent execution with query: '{query}'")
            try:
                # Define the initial state according to ReWOOState TypedDict
                initial_state = {
                    "original_query": query,
                    "plan": [], # Initialize as empty list instead of None
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

                # Create the config dictionary required by the nodes
                # The base_graph_config already contains llm, tools_str, tools, tool_names
                config = {"configurable": base_graph_config}
                # Log config keys only, excluding potentially large values like tools_str or tool objects
                logger.info(f"Config for agent execution: Configurable keys = {list(config['configurable'].keys())}")

                # Use the compiled_app object for streaming with the correct initial state and config
                async for event in compiled_app.astream_events(
                    initial_state,
                    config=config,
                    version="v1",
                ):
                    kind = event['event']
                    if kind == "on_chat_model_stream":
                        content = event["data"]["chunk"].content
                        if content:
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
                        logger.info("-" * 40)
                logger.info("Agent execution stream finished.")
            except Exception as e:
                logger.error(f"Error during agent execution: {e}", exc_info=True)

    except Exception as e:
        logger.error(f"Failed to initialize or use MultiServerMCPClient: {e}", exc_info=True)
        return # Exit if client fails

    logger.info("Agent main function finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ReWOO Agent with optional graph visualization.")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate a visualization of the LangGraph graph and save it as graph_visualization.png, then exit."
    )
    args = parser.parse_args()

    # Pass the visualize flag to the main function
    asyncio.run(main(visualize=args.visualize))
