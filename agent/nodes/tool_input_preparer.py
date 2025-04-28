from typing import Dict, Any, Optional
import logging
import re
import json

from ..state import ReWOOState, EvidenceDict
from langgraph.graph import END

# --- Logging Setup --- #
logger = logging.getLogger(__name__)
# --- End Logging Setup --- #

# Simple evidence substitution function for string inputs
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
    Prepares the input string for the selected tool by substituting evidence placeholders.
    Assumes tools take a single string argument as per the ReWOO paper examples.
    Sets workflow_status to route to 'tool_executor' on success, 'failed' on error.
    """
    logger.info("--- Starting Tool Input Preparation Node ---")
    current_tool_call_info = state.get("current_tool_call")
    evidence_dict = state.get("evidence", {})
    current_step_index = state.get("current_step_index", 0)
    step_number = current_step_index + 1

    # --- Defensive Checks --- #
    if not current_tool_call_info:
        logger.error(f"current_tool_call info is missing in state at step {step_number}. Cannot prepare input.")
        return {"workflow_status": "failed", "error_message": f"Missing tool call info at step {step_number}", "next_node": END}

    tool_name = current_tool_call_info.get("tool_name")
    raw_input_str = current_tool_call_info.get("raw_input")
    evidence_var = current_tool_call_info.get("evidence_var") # Needed for executor

    if tool_name is None or raw_input_str is None:
        logger.error(f"Tool name ('{tool_name}') or raw input ('{raw_input_str}') is missing in tool_call_info at step {step_number}.")
        return {"workflow_status": "failed", "error_message": f"Missing tool name or input at step {step_number}", "next_node": END}
    # --- End Defensive Checks --- #

    try:
        # Substitute evidence placeholders (#E1, #E2, ...)
        logger.debug(f"Raw input string for step {step_number}: '{raw_input_str}'")
        logger.debug(f"Evidence dictionary for substitution: {evidence_dict}")
        prepared_input_str = substitute_evidence(raw_input_str, evidence_dict, step_number)
        logger.info(f"Prepared input string for '{tool_name}' (Step {step_number}): '{prepared_input_str}'")

        # Prepare the final input based on the string
        if prepared_input_str == '':
            prepared_input = {}
            logger.info(f"Converted empty string input to empty dictionary {{}} for tool '{tool_name}'.")
        else:
            # Try parsing the string as JSON
            try:
                prepared_input = json.loads(prepared_input_str)
                # Ensure the result is a dictionary if parsing succeeds
                if isinstance(prepared_input, dict):
                    logger.info(f"Successfully parsed input string as JSON dictionary for tool '{tool_name}'.")
                else:
                    # If parsing gives a non-dict (e.g., list, string, number), revert to string
                    logger.warning(f"Parsed input string as JSON, but result is not a dictionary ({type(prepared_input)}). Reverting to string input for tool '{tool_name}'.")
                    prepared_input = prepared_input_str
            except json.JSONDecodeError:
                # If JSON parsing fails, assume it's a regular string input
                logger.info(f"Input string is not valid JSON. Treating as plain string input for tool '{tool_name}'.")
                prepared_input = prepared_input_str

        # Store the prepared input and indicate completion
        return {
            "prepared_tool_input": prepared_input,
            "workflow_status": "routing_complete",
            "next_node": "tool_executor",
            "error_message": None # Clear any previous errors if successful
        }
    except Exception as e:
        logger.error(f"Error substituting arguments for tool call at step {step_number}", exc_info=True)
        return {
            "workflow_status": "failed",
            "error_message": f"Failed to substitute arguments for step {step_number}: {e}",
            "next_node": END
        } 