# agent/graph.py
from typing import List, Dict, Any, Optional, Tuple, cast
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END, START, MessagesState
from functools import partial
import logging
from langsmith import traceable
from langchain_core.runnables import RunnableConfig

from .state import ReWOOState # PlanStep, Evidence, ToolResult, ToolCall 등은 State 또는 노드에서 사용

from .utils import format_tool_descriptions_with_schema, format_tool_descriptions_simplified
from langchain_community.llms import LlamaCpp
from langchain_mcp_adapters.client import MultiServerMCPClient

from .nodes.tool_filter_node import tool_filter_node
from .nodes.planner import planning_node
from .nodes.plan_parser import plan_parser_node
from .nodes.plan_validator import plan_validator_node
from .nodes.tool_selector import tool_selection_node
from .nodes.tool_input_preparer import tool_input_preparation_node
from .nodes.tool_executor import tool_execution_node
from .nodes.evidence_processor import evidence_processor_node
from .nodes.final_answer import final_answer_node

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
    next_node_hint = state.get("next_node")
    current_step_idx = state.get("current_step_index", -1) # Get current step for logging
    plan_len = len(state.get("plan", [])) # Get plan length for logging

    # Detailed logging of the current state relevant to routing
    logger.info(
        f"--- Determining Next Node --- "
        f"Status: '{status}', Hint: '{next_node_hint}', "
        f"Current Step: {current_step_idx}, Plan Length: {plan_len}, "
        f"Original Query: '{state.get('original_query', '')[:50]}...', " # Log part of query
        f"Error: '{state.get('error_message', None)}'"
    )

    # Check for explicit routing hint first
    if next_node_hint:
        logger.info(f"Routing based on explicit next_node hint: '{next_node_hint}'.")
        return next_node_hint

    # If no hint, check for terminal states based on workflow_status
    if status == "finished":
        logger.info("Workflow finished successfully based on status. Routing to END.")
        return END
    if status == "failed":
        error_msg = state.get('error_message', 'Unknown error')
        logger.error(f"Workflow failed based on status. Error: {error_msg}. Routing to END.")
        return END
    
    # Fallback: if no hint and no explicit terminal status, implies an issue or unexpected state.
    logger.error(
        f"Unexpected state for routing: No next_node hint and workflow_status is '{status}'. "
        f"This might indicate a node isn't correctly setting routing state. Forcing END to prevent loop."
    )
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

    # Add nodes (Original set, dummy node removed)
    workflow.add_node("tool_filter", partial(tool_filter_node, node_config=base_graph_config))
    workflow.add_node("planner", partial(planning_node, node_config=base_graph_config))
    workflow.add_node("plan_parser", partial(plan_parser_node, node_config=base_graph_config))
    workflow.add_node("plan_validator", partial(plan_validator_node, node_config=base_graph_config))
    workflow.add_node("tool_selector", tool_selection_node)
    workflow.add_node("tool_input_preparer", tool_input_preparation_node)
    workflow.add_node("tool_executor", partial(tool_execution_node, node_config=base_graph_config))
    workflow.add_node("evidence_processor", evidence_processor_node)
    workflow.add_node("generate_final_answer", partial(final_answer_node, node_config=base_graph_config))

    # Define edges (Restore original flow)
    workflow.add_edge(START, "tool_filter")
    workflow.add_edge("tool_filter", "planner")
    workflow.add_edge("planner", "plan_parser")
    workflow.add_edge("plan_parser", "plan_validator")

    # Conditional edge from Plan Validator (using existing should_continue)
    workflow.add_conditional_edges(
        "plan_validator",
        should_continue,
        {
            "planner": "planner",
            "tool_selector": "tool_selector",
            "generate_final_answer": "generate_final_answer",
            END: END
        }
    )

    # *** RESTORED Conditional edge from Tool Selector ***
    workflow.add_conditional_edges(
        "tool_selector",
        should_continue, # Use original should_continue based on next_node hint from tool_selector
        {
            "tool_input_preparer": "tool_input_preparer", # If tool selected
            "generate_final_answer": "generate_final_answer", # If plan finished
            "tool_selector": "tool_selector", # Fallback for thought-only steps handled within tool_selector
            END: END # On error
        }
    )
    
    # Conditional edges from Tool Input Preparer (using existing should_continue)
    workflow.add_conditional_edges(
        "tool_input_preparer",
        should_continue,
        {
            "tool_executor": "tool_executor",
            "planner": "planner",
            END: END
        }
    )

    # Fixed edge from Tool Executor to Evidence Processor
    workflow.add_edge("tool_executor", "evidence_processor")

    # Conditional edges from Evidence Processor (using existing should_continue)
    workflow.add_conditional_edges(
        "evidence_processor",
        should_continue,
        {
            "tool_selector": "tool_selector",
            "generate_final_answer": "generate_final_answer",
            END: END
        }
    )

    # Final Answer node leads to END
    workflow.add_edge("generate_final_answer", END)

    # Compile the graph
    graph = workflow.compile()
    logger.info("ReWOO agent graph built and compiled successfully (Original Structure).") # Log message updated

    return graph, base_graph_config
