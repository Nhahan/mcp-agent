# agent/graph.py
from typing import List, Dict, Any, Optional, Tuple, cast
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END, START
from functools import partial
import asyncio
import json
import re
import logging
from langsmith import traceable
from langchain_core.runnables import RunnableConfig

# Import state definition
from .state import ReWOOState # PlanStep, Evidence, ToolResult, ToolCall 등은 State 또는 노드에서 사용

# 삭제된 import:
# from .planner import generate_plan, _format_plan_to_string, _format_tool_descriptions, PlanStep
# from .solver import generate_final_answer

# Import MCP client from adapter library
from langchain_mcp_adapters.client import MultiServerMCPClient

# Import node functions from the correct location
from .nodes.planner import planning_node
from .nodes.tool_selector import tool_selection_node
from .nodes.tool_input_preparer import tool_input_preparation_node
from .nodes.tool_executor import tool_execution_node
from .nodes.evidence_processor import evidence_processor_node
from .nodes.final_answer import final_answer_node

# 수정된 프롬프트 import (상대 경로 사용)
from .prompts.plan_prompts import PLANNER_PROMPT_TEMPLATE, PLANNER_REFINE_PROMPT_TEMPLATE
from .prompts.answer_prompts import FINAL_ANSWER_PROMPT_TEMPLATE

# 삭제된 import:
# from agent.validation import PlanValidator # Planner 노드 내부에서 import 해야 할 수 있음
# from agent.tools.registry import ToolRegistry # MCP Client가 도구 제공

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel

# --- Logging Setup --- #
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)
# --- End Logging Setup --- #

# Helper function to find a tool by name from the list (MCP Client handles this now, maybe remove?)
# def find_tool_by_name(tools: List[BaseTool], name: str) -> Optional[BaseTool]:
#     for tool in tools:
#         if tool.name == name:
#             return tool
#     return None

# --- LangGraph Nodes (Definitions are in agent/nodes/*) ---
# Make sure the node functions imported above are used in add_node calls below

# --- Conditional Edges ---
def should_continue(state: ReWOOState) -> str:
    """Determines the next node to route to based on the workflow status."""
    status = state.get("workflow_status")
    next_node_hint = state.get("next_node") # Get hint regardless of status first
    logger.debug(f"--- Determining Next Node --- State Status: {status}, Next Node Hint: {next_node_hint}")

    # Check for explicit routing first
    # Ensure the status is indeed 'routing_complete' before using the hint
    if status == "routing_complete" and next_node_hint:
        logger.info(f"Routing complete, explicit next node: '{next_node_hint}'. Returning hint.")
        return next_node_hint

    # Check for terminal states
    elif status == "finished":
        logger.info("Workflow finished successfully. Routing to END.")
        return END
    elif status == "failed":
        error_msg = state.get('error_message', 'Unknown error')
        logger.error(f"Workflow failed. Error: {error_msg}. Routing to END.")
        return END
    else:
        # If status is not routing_complete, finished, or failed, it's an unexpected situation for routing
        logger.error(f"Unexpected state encountered for routing decision (Status: '{status}', Hint: '{next_node_hint}'). Forcing END.")
        return END

    # --- Old routing logic (commented out as new logic above handles it) ---
    # if status == "planner_success":
    #     return "tool_selector"
    # elif status == "tool_input_preparation":
    #     return "tool_input_preparer"
    # elif status == "tool_execution":
    #     return "tool_executor"
    # elif status == "evidence_processing":
    #     return "evidence_processor"
    # elif status == "final_answer":
    #     return "generate_final_answer"
    # elif status == "tool_selection":
    #      # This happens after a 'thought only' step or evidence processing
    #     return "tool_selector"
    # elif status == "finished":
    #     return END
    # elif status == "failed" or status == "failed_retryable": # Treat retryable as failed for routing now
    #     logger.error(f"Workflow failed. Error: {state.get('error_message')}")
    #     return END
    # else:
    #     logger.error(f"Unknown workflow status: {status}. Ending workflow.")
    #     return END
    # --- End Old routing logic ---

# --- Build Graph ---
async def build_rewoo_graph(llm: BaseLanguageModel, mcp_client: MultiServerMCPClient) -> Tuple[StateGraph, str]:
    """Builds the LangGraph StateGraph for the ReWOO agent."""
    logger.info("Building ReWOO agent graph...")
    workflow = StateGraph(ReWOOState)

    # Add nodes using the imported functions
    # The llm and mcp_client need to be passed to the nodes that require them.
    # We can use functools.partial or pass them via config["configurable"]

    # Pass LLM to planner and final_answer nodes
    # Pass MCP client to tool_executor node
    # Pass tool descriptions string via config["configurable"] to planner
    workflow.add_node("planner", planning_node) # LLM passed via config
    workflow.add_node("tool_selector", tool_selection_node)
    workflow.add_node("tool_input_preparer", tool_input_preparation_node)
    # Pass config to tool_executor, not mcp_client directly
    workflow.add_node("tool_executor", tool_execution_node)
    workflow.add_node("evidence_processor", evidence_processor_node)
    workflow.add_node("generate_final_answer", partial(final_answer_node, llm=llm))

    # Define edges
    workflow.add_edge(START, "planner")

    # Use the single conditional edge function `should_continue`
    workflow.add_conditional_edges(
        "planner",
        should_continue,
        # Provide mapping from the hints returned by `should_continue` to actual node names
        {
            "tool_selector": "tool_selector",
            "generate_final_answer": "generate_final_answer",
            END: END
            # Add other potential routes from planner if needed
        }
    )
    workflow.add_conditional_edges(
        "tool_selector",
        should_continue,
        {
            "tool_input_preparer": "tool_input_preparer",
            "generate_final_answer": "generate_final_answer",
            "tool_selector": "tool_selector", # Loop for thought-only steps
            END: END
        }
    )
    workflow.add_conditional_edges(
        "tool_input_preparer",
        should_continue,
        {
            "tool_executor": "tool_executor",
            END: END
        }
    )
    workflow.add_conditional_edges(
        "tool_executor",
        should_continue,
        {
            "evidence_processor": "evidence_processor",
            END: END
            # Add retry logic routing here if needed
        }
    )
    workflow.add_conditional_edges(
        "evidence_processor",
        should_continue,
        {
            "tool_selector": "tool_selector", # Loop back to select next tool
            "generate_final_answer": "generate_final_answer", # Should not happen here ideally
            END: END
        }
    )

    workflow.add_edge("generate_final_answer", END)

    # Compile the graph
    logger.info("Compiling the graph...")
    app = workflow.compile()
    logger.info("Graph compiled successfully.")
    # Return the compiled app and the entry point node name (usually START)
    entry_point_node = START # Or the first node if different
    return app, entry_point_node

# Example usage (for testing individual components if needed)
# async def run_test():
#     # Setup mock LLM, MCP client, initial state etc.
#     # ...
#     app, entry_node = await build_rewoo_graph(mock_llm, mock_mcp_client)
#     # Invoke the graph
#     # ...

# if __name__ == "__main__":
#     asyncio.run(run_test()) 