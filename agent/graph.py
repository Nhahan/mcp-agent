# agent/graph.py
from typing import List, Dict, Any, Optional, Tuple, cast
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END, START, MessagesState
from functools import partial
import json
import logging
from langsmith import traceable
from langchain_core.runnables import RunnableConfig

from .state import ReWOOState # PlanStep, Evidence, ToolResult, ToolCall 등은 State 또는 노드에서 사용

from .utils import format_tool_descriptions_with_schema, format_tool_descriptions_simplified # Import both formatters
from langchain_community.llms import LlamaCpp # Import to check instance type

# Import MCP client from adapter library
from langchain_mcp_adapters.client import MultiServerMCPClient
# Removed MCPToolAdapter import

# Import node functions from the correct location
from .nodes.tool_filter_node import tool_filter_node # Import the new node
from .nodes.planner import planning_node
from .nodes.plan_parser import plan_parser_node # Added plan_parser_node import
from .nodes.plan_validator import plan_validator_node # Import the new validator node
from .nodes.tool_selector import tool_selection_node
from .nodes.tool_input_preparer import tool_input_preparation_node
from .nodes.tool_executor import tool_execution_node
from .nodes.evidence_processor import evidence_processor_node
from .nodes.final_answer import final_answer_node

# 수정된 프롬프트 import (상대 경로 사용)
from .prompts.plan_prompts import PLANNER_PROMPT_TEMPLATE, PLANNER_REFINE_PROMPT_TEMPLATE
from .prompts.answer_prompts import FINAL_ANSWER_PROMPT_TEMPLATE

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel

# --- Logging Setup --- #
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)
# --- End Logging Setup --- #


# --- Conditional Edges --- #
def should_continue(state: ReWOOState) -> str:
    """Determines the next node to route to based on the workflow status and next_node hint."""
    status = state.get("workflow_status")
    next_node_hint = state.get("next_node") # Get hint regardless of status first
    logger.debug(f"--- Determining Next Node --- State Status: {status}, Next Node Hint: {next_node_hint}")

    # Check for explicit routing hint first (most common case for successful steps)
    if next_node_hint:
        logger.info(f"Routing based on explicit next_node hint: '{next_node_hint}'.")
        # Clear the hint after using it to prevent accidental reuse on loops/retries
        # state["next_node"] = None # Modification of state dict directly is discouraged in conditional edges
        return next_node_hint

    # Check for terminal states if no hint
    elif status == "finished":
        logger.info("Workflow finished successfully. Routing to END.")
        return END
    elif status == "failed":
        error_msg = state.get('error_message', 'Unknown error')
        logger.error(f"Workflow failed. Error: {error_msg}. Routing to END.")
        return END
    else:
        # If no hint and not a terminal state, it's unexpected.
        # This might happen if a node doesn't set next_node correctly.
        logger.error(f"Unexpected state for routing: No next_node hint and status is '{status}'. Forcing END.")
        return END

# --- Build Graph --- #
def build_rewoo_graph(base_llm: BaseLanguageModel, mcp_client: MultiServerMCPClient) -> Tuple[StateGraph, Dict[str, Any]]: # Changed first arg name
    """Builds the LangGraph StateGraph for the ReWOO agent."""
    logger.info("Building ReWOO agent graph...")

    # --- Get Tools (Full List) --- #
    logger.info("Fetching ALL tools from MCP client...")
    all_tools: List[BaseTool] = mcp_client.get_tools()
    
    # -----> LOGGING: Initial tool names from client <-----
    initial_tool_names = [t.name for t in all_tools] if all_tools else []
    logger.info(f"INITIAL tool names received from MCP client: {initial_tool_names}")
    # ---------------------------------------------------
    
    simplified_tools_str = "No tools available."
    full_schema_tools_str = "No tools available." # For grammar generation if needed
    if not all_tools:
        logger.warning("No tools discovered via MCP client.")
    else:
        simplified_tools_str = format_tool_descriptions_simplified(all_tools)
        # Generate full schema descriptions needed for GBNF generation
        full_schema_tools_str = format_tool_descriptions_with_schema(all_tools)
    # --- End Tool Fetching & Formatting --- #
    
    # --- Prepare Planner-Specific LLM (REMOVED GBNF) --- #
    planner_llm = base_llm # Planner now always uses the base LLM without grammar
    logger.info("GBNF is disabled. Planner will use the base LLM instance directly.")
    # --- End Planner LLM Prep --- #

    workflow = StateGraph(ReWOOState)

    # --- Prepare Base Config --- #
    base_graph_config = {
        "configurable": {
            "llm": base_llm,
            "planner_llm": planner_llm, # Now same as base_llm
            "tools": all_tools,
            "tool_names": [t.name for t in all_tools],
            # Pass SIMPLIFIED descriptions for the filter node
            "simplified_tool_descriptions": simplified_tools_str,
            # Pass FULL schema descriptions for potential use in other nodes if needed
            "full_schema_tool_descriptions": full_schema_tools_str
        }
    }
    logger.debug(f"Base graph config created. Available tools: {[t.name for t in base_graph_config['configurable']['tools']]}")

    # Add nodes (Parser node needs config now for LLM call)
    workflow.add_node("tool_filter", partial(tool_filter_node, node_config=base_graph_config))
    workflow.add_node("planner", partial(planning_node, node_config=base_graph_config))
    workflow.add_node("plan_parser", partial(plan_parser_node, node_config=base_graph_config)) # Pass config to parser
    workflow.add_node("plan_validator", partial(plan_validator_node, node_config=base_graph_config)) # Pass config to validator
    workflow.add_node("tool_selector", tool_selection_node)
    workflow.add_node("tool_input_preparer", tool_input_preparation_node)
    workflow.add_node("tool_executor", partial(tool_execution_node, node_config=base_graph_config))
    workflow.add_node("evidence_processor", evidence_processor_node)
    workflow.add_node("generate_final_answer", partial(final_answer_node, node_config=base_graph_config))

    # Define edges
    workflow.add_edge(START, "tool_filter") # Start with filtering
    workflow.add_edge("tool_filter", "planner") # Filter output goes to planner
    workflow.add_edge("planner", "plan_parser") # Planner output goes to the parser (which might call LLM)
    workflow.add_edge("plan_parser", "plan_validator") # Parser (potentially corrected) output goes to validator

    # Conditional edges from Plan Validator
    workflow.add_conditional_edges(
        "plan_validator",
        should_continue, # Re-use the same routing logic based on next_node hint
        {
            "planner": "planner", # If validation fails and retry needed
            "tool_selector": "tool_selector", # If validation succeeds and plan has tools
            "generate_final_answer": "generate_final_answer", # If validation succeeds and plan has no tools
            END: END # If validation fails and max retries reached or critical error
        }
    )

    # Conditional edges from Tool Selector
    workflow.add_conditional_edges(
        "tool_selector",
        should_continue,
        {
            "tool_input_preparer": "tool_input_preparer", # If tool selected
            "generate_final_answer": "generate_final_answer", # If no more steps/tools
            "tool_selector": "tool_selector", # Should not happen if logic is correct, but acts as fallback
            END: END
        }
    )

    # Conditional edges from Tool Input Preparer
    workflow.add_conditional_edges(
        "tool_input_preparer",
        should_continue,
        {
            "tool_executor": "tool_executor", # If input prep succeeds
            "planner": "planner", # If input prep fails (e.g., missing evidence), retry planning
            END: END # If critical error during input prep
        }
    )

    workflow.add_edge("tool_executor", "evidence_processor")

    # Conditional edges from Evidence Processor
    workflow.add_conditional_edges(
        "evidence_processor",
        should_continue,
        {
            "tool_selector": "tool_selector", # Loop back to select next tool/step
            "generate_final_answer": "generate_final_answer", # If all steps done
            END: END # If error during evidence processing
        }
    )

    # Final Answer node leads to END
    workflow.add_edge("generate_final_answer", END)

    # Compile the graph
    graph = workflow.compile()
    logger.info("ReWOO agent graph built and compiled successfully (GBNF Disabled).")

    return graph, base_graph_config
