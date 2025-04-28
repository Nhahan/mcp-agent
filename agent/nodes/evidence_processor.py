# agent/nodes/evidence_processor.py
from typing import Dict, Any
import logging

from ..state import ReWOOState

# --- Logging Setup --- #
logger = logging.getLogger(__name__)
# --- End Logging Setup --- #


async def evidence_processor_node(state: ReWOOState) -> Dict[str, Any]:
    """
    Processes the latest piece of evidence (e.g., summarize, extract info).
    Placeholder: Currently just passes the evidence through.
    Increments the step index and sets status to 'tool_selection'.
    """
    logger.info("--- Starting Evidence Processor Node --- (Placeholder)")
    # Get the current state information
    current_step_index = state.get("current_step_index", 0) # This is the *next* step index after tool execution
    plan = state.get("plan")
    evidence_dict = state.get("evidence", {}) # This is the DICT mapping #E<n> to results

    # --- Corrected Logic ---
    completed_step_index = current_step_index - 1 # Index of the step just finished
    if completed_step_index < 0:
         logger.warning("Evidence processor called before any step was completed.")
         return {"workflow_status": "routing_complete", "next_node": "tool_selector"} # Or handle appropriately

    # Construct the evidence key for the completed step (e.g., #E1 for index 0)
    evidence_var_key = f"#E{completed_step_index + 1}"

    # Safely get the evidence from the dictionary
    latest_evidence = evidence_dict.get(evidence_var_key)

    if latest_evidence is None:
        logger.warning(f"Evidence for key '{evidence_var_key}' not found in evidence dictionary: {evidence_dict}")
        # Decide how to proceed - maybe skip processing? Route back?
        # For now, just route back to tool_selector
        return {"workflow_status": "routing_complete", "next_node": "tool_selector"}

    logger.info(f"Retrieved evidence for step {completed_step_index + 1} ('{evidence_var_key}'): {str(latest_evidence)[:200]}...")

    # Placeholder: Actual processing logic (e.g., LLM summarization) would go here.
    # For now, we just use the raw evidence fetched.
    # processed_info = f"Processed: {str(latest_evidence)[:100]}..."
    # logger.info("Evidence processing complete (Placeholder).")

    # No state update needed here as we are just using the raw evidence for now.
    # If we were creating processed_info, we might update the dict:
    # evidence_dict[evidence_var_key] = processed_info # Or use a different key structure

    logger.debug(f"Evidence dictionary remains: {evidence_dict}")
    # --- End Corrected Logic ---


    # Update state (This logic needs to be revised based on how evidence is structured)
    # Example: update the evidence dict with processed info if needed
    # evidence_dict[f"#E{current_step_index}"] = processed_info # This assumes #E key matches step index

    #logger.debug(f"Evidence dictionary after processing attempt: {evidence_dict}")

    return {
        "evidence": evidence_dict, # Return the potentially updated dictionary
        "workflow_status": "routing_complete",
        "next_node": "tool_selector" # Route back to select the next tool/step
    } 