# agent/nodes/evidence_processor.py
from typing import Dict, Any
import logging
import json

from ..state import ReWOOState

logger = logging.getLogger(__name__)


async def evidence_processor_node(state: ReWOOState) -> Dict[str, Any]:
    """
    Processes the latest piece of raw evidence based on its type.
    Converts dictionaries and lists to formatted strings.
    Looks for common keys in dictionaries to extract primary content.
    Overwrites the original evidence with the processed string version.
    Routes back to tool_selector.
    """
    logger.info("--- Starting Evidence Processor Node ---")
    current_step_index = state.get("current_step_index", 0) # Index of the step *to be* selected next
    evidence_dict = state.get("evidence", {}).copy() # Use a copy to modify

    # Index of the step that just finished and produced the evidence
    completed_step_index = current_step_index - 1 
    if completed_step_index < 0:
         logger.warning("Evidence processor called before any step was completed.")
         # Still need to return current_step_index for state consistency
         return {
             "evidence": evidence_dict,
             "workflow_status": "routing_complete", 
             "next_node": "tool_selector",
             "current_step_index": current_step_index
        }

    # Construct the evidence key for the completed step (e.g., #E1 for index 0)
    evidence_var_key = f"#E{completed_step_index + 1}"

    # Get the raw evidence object from the dictionary
    raw_evidence = evidence_dict.get(evidence_var_key)

    if raw_evidence is None:
        logger.warning(f"Raw evidence for key '{evidence_var_key}' not found in evidence dictionary: {evidence_dict}")
        processed_evidence_str = f"Error: Evidence for {evidence_var_key} not found."
    else:
        logger.info(f"Processing raw evidence for step {completed_step_index + 1} ('{evidence_var_key}'), type: {type(raw_evidence)}")
        # Process based on type
        if isinstance(raw_evidence, str):
            processed_evidence_str = raw_evidence
        elif isinstance(raw_evidence, dict):
            # Try common keys first
            common_keys = ["output", "result", "content", "thought", "summary", "answer"]
            found_key = None
            for key in common_keys:
                if key in raw_evidence:
                    processed_evidence_str = str(raw_evidence[key]) # Convert extracted value to string
                    found_key = key
                    break
            if found_key:
                 logger.info(f"Extracted value from dictionary using key: '{found_key}'")
            else:
                 # If no common key, pretty-print the dictionary
                 try:
                     processed_evidence_str = json.dumps(raw_evidence, indent=2)
                     logger.info("No common key found, converted dictionary to JSON string.")
                 except TypeError as e:
                     logger.warning(f"Could not JSON serialize dictionary evidence for {evidence_var_key}, falling back to str(): {e}")
                     processed_evidence_str = str(raw_evidence)
        elif isinstance(raw_evidence, list):
            # Convert list to a simple string representation (e.g., newline separated)
            try:
                 processed_evidence_str = "\n".join([str(item) for item in raw_evidence])
                 logger.info("Converted list evidence to newline-separated string.")
            except Exception as e:
                 logger.warning(f"Could not process list evidence for {evidence_var_key}, falling back to str(): {e}")
                 processed_evidence_str = str(raw_evidence)
        else:
            # Fallback for other types
            processed_evidence_str = str(raw_evidence)
            logger.info(f"Converted evidence of type {type(raw_evidence)} to string.")
            
    # Overwrite the evidence dictionary with the processed string
    evidence_dict[evidence_var_key] = processed_evidence_str
    logger.info(f"Stored processed evidence for {evidence_var_key}: {processed_evidence_str[:200]}...")
    logger.debug(f"Evidence dictionary after processing: {evidence_dict}")
    
    return {
        "evidence": evidence_dict, # Return the updated dictionary with processed string
        "workflow_status": "routing_complete",
        "next_node": "tool_selector", # Route back to select the next tool/step
        "current_step_index": current_step_index # Pass the index along
    } 