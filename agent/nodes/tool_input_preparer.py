from typing import Dict, Any
import logging
import re
import json

from ..state import ReWOOState, EvidenceDict
from langgraph.graph import END

logger = logging.getLogger(__name__)


def substitute_evidence(input_str: str, evidence: EvidenceDict, step_number_for_log: int) -> str:
    """ Substitutes #E<n> placeholders in the input string with evidence. """
    # Pattern to find all #E<n> placeholders
    pattern = re.compile(r"(#E\d+)")

    def replace_match(match):
        placeholder = match.group(1)
        if placeholder in evidence:
            logger.debug(f"(Step {step_number_for_log}) Substituting placeholder '{placeholder}' with evidence.")
            # Ensure evidence is string, as we are substituting into a string
            return str(evidence[placeholder])
        else:
            logger.warning(f"(Step {step_number_for_log}) Evidence placeholder '{placeholder}' not found in evidence dict: {list(evidence.keys())}. Keeping placeholder.")
            return placeholder # Keep placeholder if not found

    substituted_str = pattern.sub(replace_match, input_str)
    return substituted_str


async def tool_input_preparation_node(state: ReWOOState) -> Dict[str, Any]:
    """
    Prepares the input arguments for the selected tool by substituting evidence placeholders
    in each argument value if it's a string. 
    The input arguments are expected to be a dictionary.
    Sets workflow_status to route to 'tool_executor' on success, 'failed' on error.
    """
    logger.info("--- Starting Tool Input Preparation Node ---")
    current_tool_call_info = state.get("current_tool_call")
    evidence_dict = state.get("evidence", {})
    current_step_index = state.get("current_step_index", 0)
    step_number = current_step_index + 1

    if not current_tool_call_info:
        logger.error(f"current_tool_call info is missing in state at step {step_number}. Cannot prepare input.")
        return {"workflow_status": "failed", "error_message": f"Missing tool call info at step {step_number}", "next_node": END}

    tool_name = current_tool_call_info.get("tool_name")
    # Get the arguments dictionary from current_tool_call_info
    raw_arguments_dict = current_tool_call_info.get("arguments") 

    if tool_name is None or raw_arguments_dict is None: # Check if arguments dict is None
        logger.error(f"Tool name ('{tool_name}') or arguments dictionary ('{raw_arguments_dict}') is missing in tool_call_info at step {step_number}.")
        return {"workflow_status": "failed", "error_message": f"Missing tool name or arguments dict at step {step_number}", "next_node": END}
    
    if not isinstance(raw_arguments_dict, dict):
        logger.error(f"Tool arguments are not a dictionary ('{type(raw_arguments_dict)}') for tool '{tool_name}' at step {step_number}.")
        return {"workflow_status": "failed", "error_message": f"Tool arguments not a dict for {tool_name} at step {step_number}", "next_node": END}

    try:
        prepared_arguments = {}
        logger.debug(f"Raw arguments dictionary for step {step_number}: {raw_arguments_dict}")
        logger.debug(f"Evidence dictionary for substitution: {evidence_dict}")

        for key, value in raw_arguments_dict.items():
            if isinstance(value, str):
                # Substitute evidence placeholders only if the value is a string
                prepared_arguments[key] = substitute_evidence(value, evidence_dict, step_number)
            else:
                # If not a string, keep the original value (e.g., boolean, number, list, dict)
                prepared_arguments[key] = value
        
        logger.info(f"Prepared arguments for '{tool_name}' (Step {step_number}): {prepared_arguments}")

        return {
            "prepared_tool_input": prepared_arguments, # This is now a dictionary
            "workflow_status": "routing_complete",
            "next_node": "tool_executor",
            "error_message": None
        }
    except Exception as e:
        logger.error(f"Error preparing arguments for tool '{tool_name}' at step {step_number}", exc_info=True)
        return {
            "workflow_status": "failed",
            "error_message": f"Failed to prepare arguments for {tool_name} at step {step_number}: {e}",
            "next_node": END
        } 