# agent/nodes/tool_selector.py
from typing import Dict, Any
import logging

from ..state import ReWOOState
from langgraph.graph import END

# --- Logging Setup --- #
logger = logging.getLogger(__name__)
# --- End Logging Setup --- #


async def tool_selection_node(state: ReWOOState) -> Dict[str, Any]:
    """
    Selects the tool for the current step based on the parsed plan tuple.
    Routes to final answer generation if the plan is complete or an error occurred.
    Updates workflow_status to route to 'tool_input_preparer', 'generate_final_answer', or loops back.
    """
    logger.info("--- Starting Tool Selection Node ---")

    plan = state.get("plan")
    current_step_index = state.get("current_step_index", 0)

    if not plan:
        logger.error("Tool selector called with empty plan. Routing to END.")
        return {"workflow_status": "failed", "error_message": "Plan is empty.", "next_node": END}

    if current_step_index >= len(plan):
        logger.info("Plan finished. Moving to final answer generation.")
        return {"workflow_status": "routing_complete", "next_node": "generate_final_answer"}

    # Access the current step tuple
    current_step_tuple = plan[current_step_index]
    step_number = current_step_index + 1

    # Extract information using tuple indices
    try:
        reasoning = current_step_tuple[0]
        evidence_var = current_step_tuple[1]
        tool_name = current_step_tuple[2]
        tool_input_str = current_step_tuple[3] # Input is initially a string
    except (IndexError, TypeError) as e:
        logger.error(f"Error accessing elements from plan step tuple at index {current_step_index}: {current_step_tuple}. Error: {e}")
        return {
            "workflow_status": "failed",
            "error_message": f"Invalid plan step format at index {current_step_index}.",
            "next_node": END
        }

    # Check if a tool name is present (index 2)
    if not tool_name:
        # This case might be less common now if the parser ensures a tool name,
        # but handle it defensively. Treat as a thought-only step.
        logger.info(f"No tool name found for step {step_number}. Treating as thought-only step.")
        thought_evidence_str = f"Step {step_number} Thought: {reasoning}"
        logger.debug(f"Adding thought as evidence: {thought_evidence_str}")
        # Update evidence dictionary directly
        current_evidence_dict = state.get("evidence", {})
        current_evidence_dict[evidence_var] = thought_evidence_str # Use evidence_var as key
        return {
            "current_step_index": current_step_index + 1,
            "evidence": current_evidence_dict,
            "current_tool_call": None, # No tool call info
            "workflow_status": "routing_complete",
            "next_node": "tool_selector" # Loop back to TOOL_SELECTOR for next step
        }

    logger.info(f"Selected tool: '{tool_name}' for step {step_number}")
    # Prepare the current tool call information for the next node (input preparer)
    # The tool_input_str needs further processing in the next node to handle #E variable substitution
    current_tool_call_info = {
        "tool_name": tool_name,
        "raw_input": tool_input_str, # Pass the raw input string
        "evidence_var": evidence_var # Pass the variable name for evidence storage
    }
    return {
        "tool_name": tool_name, # Still useful for routing/logging maybe?
        "current_tool_call": current_tool_call_info,
        "workflow_status": "routing_complete",
        "next_node": "tool_input_preparer"
    } 