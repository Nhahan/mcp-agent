# agent/nodes/tool_selector.py
from typing import Dict, Any, List, Optional
import logging

from ..state import ReWOOState
from langgraph.graph import END
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

# Helper function (copied from tool_executor) - needed if tools are not in state
# Alternatively, ensure filtered_tools list is always available in state for this node
def find_tool_by_name(tools: List[BaseTool], name: str) -> Optional[BaseTool]:
    for tool in tools:
        if tool.name == name:
            return tool
    return None


async def tool_selection_node(state: ReWOOState) -> Dict[str, Any]:
    """
    Selects the tool for the current step based on the parsed plan (all_parsed_steps list of dicts).
    Checks tool metadata (evidence_source_key) to determine if an input arg should be evidence.
    Routes to final answer generation if the plan is complete or an error occurred.
    Updates workflow_status to route to 'tool_input_preparer', 'generate_final_answer', or loops back.
    """
    logger.info("--- Starting Tool Selection Node ---")

    parsed_plan_steps = state.get("all_parsed_steps") 
    current_step_index = state.get("current_step_index", 0)
    # Get the filtered tools list from the state - IMPORTANT: Tool Filter node must populate this
    filtered_tools_list = state.get("filtered_tools", [])

    if not parsed_plan_steps:
        logger.error("Tool selector called with empty parsed_plan_steps. Routing to END.")
        return {"workflow_status": "failed", "error_message": "Parsed plan (all_parsed_steps) is empty.", "next_node": END}

    if current_step_index >= len(parsed_plan_steps):
        logger.info("Plan finished. Moving to final answer generation.")
        return {"workflow_status": "routing_complete", "next_node": "generate_final_answer"}

    current_step_dict = parsed_plan_steps[current_step_index]
    step_number = current_step_index + 1

    reasoning = current_step_dict.get("plan")
    tool_call_info_from_plan = current_step_dict.get("tool_call")

    if not tool_call_info_from_plan:
        logger.info(f"No tool_call found for step {step_number} ('{reasoning}'). Treating as thought-only step.")
        thought_evidence_str = f"Step {step_number} Thought: {reasoning}"
        logger.debug(f"Adding thought as evidence: {thought_evidence_str}")
        
        current_evidence_dict = state.get("evidence", {})
        # Need an evidence variable for thought-only steps. 
        # The plan_parser creates steps like: {"step_index": i + 1, "plan": step.plan, "tool_call": None, "status": "pending"}
        # It doesn't assign an evidence_variable if tool_call is None.
        # We need a convention here, e.g., #T{step_number} or ensure planner assigns one even for thought steps.
        # For now, let's assume a default or skip if no evidence_var implicitly defined.
        # This part needs to be robust. Let's assume planner/parser ensures an evidence_var or we derive one.
        # Let's use a placeholder for now if tool_call_info_from_plan is None.
        # A better approach is for the planner to ensure evidence_variable for all steps, even thought-only.
        # For now, we won't store evidence if there's no explicit evidence_variable from the plan for a thought step.
        # This means thoughts are just part of the reasoning flow unless planner explicitly makes them evidence.

        return {
            "current_step_index": current_step_index + 1,
            # "evidence": current_evidence_dict, # Only update if we have an evidence_var
            "current_tool_call": None, # No tool call info
            "workflow_status": "routing_complete",
            "next_node": "tool_selector" # Loop back to TOOL_SELECTOR for next step
        }

    tool_name = tool_call_info_from_plan.get("tool_name")
    tool_arguments = tool_call_info_from_plan.get("arguments", {})
    evidence_var = tool_call_info_from_plan.get("evidence_variable")
    # Get evidence_input_key potentially set by the planner (LLM)
    evidence_input_key_from_planner = tool_call_info_from_plan.get("evidence_input_key") 
    evidence_input_key_final = evidence_input_key_from_planner # Start with planner's value

    if not tool_name or not evidence_var:
        logger.error(f"Tool name or evidence_variable missing in tool_call_info for step {step_number}. Info: {tool_call_info_from_plan}")
        return {
            "workflow_status": "failed",
            "error_message": f"Malformed tool_call_info at step {step_number}.",
            "next_node": END
        }

    # --- Check Tool Metadata for evidence_source_key --- #
    tool_object = find_tool_by_name(filtered_tools_list, tool_name)
    if not tool_object:
        # This case should ideally be caught by plan_validator, but double-check
        logger.error(f"Tool '{tool_name}' selected in step {step_number} but not found in filtered_tools list: {[t.name for t in filtered_tools_list]}")
        return {
            "workflow_status": "failed",
            "error_message": f"Selected tool '{tool_name}' not found in filtered list at step {step_number}.",
            "next_node": END
        }
        
    # Check for the metadata attribute on the tool object
    metadata_evidence_key = getattr(tool_object, 'evidence_source_key', None)
    if metadata_evidence_key and isinstance(metadata_evidence_key, str):
        logger.info(f"Tool '{tool_name}' metadata specifies using input key '{metadata_evidence_key}' as evidence.")
        if evidence_input_key_final and evidence_input_key_final != metadata_evidence_key:
             logger.warning(f"Planner specified '{evidence_input_key_final}' but tool metadata specifies '{metadata_evidence_key}'. Overriding with tool metadata.")
        evidence_input_key_final = metadata_evidence_key # Tool metadata takes precedence
    # --- End Metadata Check --- #

    logger.info(f"Selected tool: '{tool_name}' for step {step_number}. Evidence var: '{evidence_var}'. Evidence input key: '{evidence_input_key_final}'.")
    
    current_tool_call_for_state = {
        "tool_name": tool_name,
        "arguments": tool_arguments,
        "evidence_var": evidence_var,
    }
    # Use the potentially overridden key from metadata
    if evidence_input_key_final: 
        current_tool_call_for_state["evidence_input_key"] = evidence_input_key_final
        
    return {
        "current_tool_call": current_tool_call_for_state,
        "workflow_status": "routing_complete",
        "next_node": "tool_input_preparer"
    }