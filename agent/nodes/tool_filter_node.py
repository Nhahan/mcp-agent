import logging
from typing import Dict, Any, List
from langgraph.graph import END
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
import json

from ..state import ReWOOState
from ..prompts.tool_filter_prompts import TOOL_FILTER_PROMPT_TEMPLATE
from ..utils import format_tool_descriptions_with_schema # Import from utils

logger = logging.getLogger(__name__)

async def tool_filter_node(state: ReWOOState, node_config: Dict[str, Any]) -> Dict[str, Any]:
    """Filters the available tools based on the user query."""
    logger.info("--- Entering Tool Filter Node ---")
    
    query = state["original_query"]
    configurable = node_config.get("configurable", {})
    llm = configurable.get("llm") 
    all_tools: List[BaseTool] = configurable.get("tools", []) # Get the full list of tools
    
    # -----> LOGGING: Log initial tool names <-----
    initial_tool_names = [t.name for t in all_tools]
    logger.info(f"Received {len(initial_tool_names)} tools from MCP client: {initial_tool_names}")
    # ---------------------------------------------

    # Get the SIMPLIFIED tool descriptions string from config
    simplified_tools_str = configurable.get("simplified_tool_descriptions", "No tools available.")
    
    if not llm:
        logger.error("Base LLM ('llm') missing in config for tool_filter_node")
        return {"error_message": "Tool Filter Error: LLM not configured.", "workflow_status": "failed", "next_node": END}
        
    if not all_tools or simplified_tools_str == "No tools available.":
        logger.warning("No tools or simplified descriptions available to filter.")
        return {
            "filtered_tools": [],
            "filtered_tools_str": "No tools available.",
            "workflow_status": "routing_complete", # Allow planning even if no tools
            "next_node": "planner"
        }

    # Use the SIMPLIFIED descriptions for the filter prompt
    prompt_args = {
        "query": query,
        "tool_descriptions": simplified_tools_str 
    }

    filtered_tools = [] # Initialize here
    filtered_tools_str = "No tools available." # Initialize here

    try:
        logger.info("Invoking LLM to select relevant tools using SIMPLIFIED descriptions...")
        tool_filter_chain = TOOL_FILTER_PROMPT_TEMPLATE | llm
        response = await tool_filter_chain.ainvoke(prompt_args, config=node_config)
        raw_response_content = response.content if hasattr(response, 'content') else str(response)
        
        cleaned_json_str = raw_response_content.strip()
        # Basic cleaning: remove potential markdown backticks
        if cleaned_json_str.startswith("```json"):
            cleaned_json_str = cleaned_json_str[7:]
        if cleaned_json_str.startswith("```"):
             cleaned_json_str = cleaned_json_str[3:]
        if cleaned_json_str.endswith("```"):
             cleaned_json_str = cleaned_json_str[:-3]
        cleaned_json_str = cleaned_json_str.strip()

        selected_names = set() # Initialize as empty set
        try:
            # Parse the cleaned string as JSON
            parsed_list = json.loads(cleaned_json_str)
            if isinstance(parsed_list, list):
                # Ensure all items are strings and strip whitespace
                selected_names = {str(name).strip() for name in parsed_list if isinstance(name, str) and name.strip()}
            else:
                logger.warning(f"LLM output parsed as JSON, but was not a list: {type(parsed_list)}. Treating as no tools selected.")
        except json.JSONDecodeError as json_err:
            logger.warning(f"Failed to parse LLM response as JSON: {json_err}. Response was: '{cleaned_json_str}'. Attempting fallback extraction.")
            # Fallback: Try simple comma split on the cleaned string if JSON fails
            selected_names = {name.strip() for name in cleaned_json_str.split(',') if name.strip() and '/' in name} # Basic heuristic
            if selected_names:
                 logger.info(f"Fallback extraction selected names: {selected_names}") # Log fallback selected names
            else:
                 logger.warning("Fallback extraction failed to find potential tool names.")
        
        # Filter tools based on extracted names (from JSON or fallback)
        if not selected_names:
            logger.info("No relevant tool names extracted by LLM/Fallback.")
            filtered_tools = []
        else:
            logger.info(f"Filtering tools based on extracted names: {selected_names}")
            all_tools_dict = {tool.name: tool for tool in all_tools}
            # Get the actual BaseTool objects for the selected names
            filtered_tools = [all_tools_dict[name] for name in selected_names if name in all_tools_dict]
            
            # -----> LOGGING: Log the names of the final filtered tools <-----
            final_filtered_tool_names = [t.name for t in filtered_tools]
            logger.info(f"Final list of FILTERED tool names: {final_filtered_tool_names}")
            # --------------------------------------------------------------
            
            if not filtered_tools:
                 # Log if names were extracted but didn't match available tools
                 if selected_names:
                     logger.warning(f"LLM/Fallback extracted names ({selected_names}), but none matched available tool names: {[t.name for t in all_tools]}. Proceeding without tools.")
                 else: # Should not happen if selected_names is empty, but for completeness
                     logger.warning("Proceeding without tools (no names extracted).")
            else:
                 logger.info(f"Selected {len(filtered_tools)} tools: {[t.name for t in filtered_tools]}")

        # Format FULL descriptions (with schema) ONLY for the filtered tools
        if filtered_tools:
            filtered_tools_str = format_tool_descriptions_with_schema(filtered_tools)
            # -----> LOGGING: Log the descriptions being passed to the planner <-----
            logger.info(f"Formatted descriptions passed to planner:\n---\n{filtered_tools_str}\n---")
            # ----------------------------------------------------------------------
        else:
             filtered_tools_str = "No relevant tools were selected."
        
        return {
            "filtered_tools": filtered_tools, # Store the list of BaseTool objects
            "filtered_tools_str": filtered_tools_str, # Store the formatted FULL descriptions
            "workflow_status": "routing_complete",
            "error_message": None,
            "next_node": "planner" 
        }

    except Exception as e:
        logger.error(f"Error during tool filtering: {e}", exc_info=True)
        # Fallback: If filtering fails, pass all tools to the planner
        logger.warning("Tool filtering failed. Passing ALL tools with FULL descriptions to the planner as a fallback.")
        # Generate full descriptions for ALL tools in fallback case
        all_tools_full_str = format_tool_descriptions_with_schema(all_tools) 
        # -----> LOGGING: Log fallback descriptions <-----
        logger.warning(f"Fallback: Passing descriptions for ALL tools to planner:\n---\n{all_tools_full_str}\n---")
        # --------------------------------------------
        return {
            "filtered_tools": all_tools,
            "filtered_tools_str": all_tools_full_str,
            "workflow_status": "routing_complete",
            "error_message": f"Tool filtering failed: {e}", # Log error but continue
            "next_node": "planner"
        } 