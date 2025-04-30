from typing import List, Dict, Any
import logging
from langsmith import traceable
from langchain_core.runnables import RunnableConfig
import json
import yaml # Import yaml
from ..state import ReWOOState
from ..prompts.plan_prompts import PLANNER_PROMPT_TEMPLATE, PLANNER_REFINE_PROMPT_TEMPLATE


# --- Placeholder for PlanValidator (Keep as is for now) --- #
class PlanValidator:
    def validate(self, plan_dicts: List[Dict[str, Any]]) -> tuple[bool, list]:
        # Basic placeholder validation
        if not isinstance(plan_dicts, list):
            return False, [(-1, "Plan is not a list")]
        if not plan_dicts:
             return False, [(-1, "Plan cannot be empty")]
        errors = []
        for i, step in enumerate(plan_dicts):
            if not isinstance(step, dict):
                errors.append((i, f"Step {i+1} is not a dictionary"))
                continue
            if "thought" not in step or not step["thought"]:
                errors.append((i, f"Step {i+1} is missing a 'thought'"))
            if "expected_outcome" not in step or not step["expected_outcome"]:
                errors.append((i, f"Step {i+1} is missing an 'expected_outcome'"))
            if "tool_call" not in step or not isinstance(step.get("tool_call"), dict):
                errors.append((i, f"Step {i+1} has invalid tool_call structure"))
            # Add more checks (tool_call format, etc.) if needed
        return not errors, errors
# --- End Placeholder for PlanValidator --- #

from langgraph.graph import END

logger = logging.getLogger(__name__)


@traceable(name="Planning Node")
async def planning_node(state: ReWOOState, node_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates or refines a plan using the LLM, requesting YAML-like output.
    Passes the raw output to the plan_parser node.
    Sets workflow_status to 'planning_failed' on LLM error after retries.
    """
    current_retry = state.get('current_retry', 0)
    logger.info(f"--- Entering Planning Node (Attempt {current_retry + 1}) ---")
    max_retries = state.get('max_retries', 1)
    # llm_retries removed, handled by OutputFixingParser previously

    if current_retry >= max_retries:
        logger.error(f"Max retries ({max_retries}) reached for planning node LLM invocation.")
        # Keep error message simple, parser node handles detailed parsing errors
        return {"workflow_status": "failed", "error_message": f"Planning failed after {max_retries} LLM retries.", "current_retry": current_retry, "next_node": END}

    logger.info(f"Generating plan for query: '{state['original_query']}'")
    logger.debug(f"Planning Node received state keys: {list(state.keys())}")

    # Use the passed node_config dictionary
    configurable = node_config.get("configurable", {}) 
    # Use the potentially grammar-bound planner_llm from config
    llm = configurable.get("planner_llm") 
    if not llm:
        # Fallback to base llm if planner_llm is somehow not set
        logger.warning("'planner_llm' not found in config, falling back to base 'llm'.")
        llm = configurable.get("llm")
    
    # Get the filtered tools string from the state, not the config
    filtered_tools_str = state.get("filtered_tools_str") 
    if not filtered_tools_str:
        logger.warning("Filtered tools string is missing or empty in state. Planner will operate without tool descriptions.")
        # Optionally handle this case - maybe skip planning tools? For now, pass empty string.
        filtered_tools_str = "No tools available for planning."

    # === Log the actual tools string being used by the planner ===
    logger.debug(f"Planner using the following tool descriptions:\n---\n{filtered_tools_str}\n---")
    # ============================================================

    query = state["original_query"]
    last_error = state.get('error_message')
    previous_plan_pydantic = state.get('plan_pydantic')
    raw_llm_response_for_log = ""

    # Check if LLM is None before proceeding
    if not llm:
        logger.error(f"LLM (base or planner) missing in config['configurable']: {configurable}")
        return {"error_message": "LLM missing in config", "workflow_status": "failed", "current_retry": current_retry, "next_node": END}

    try:
        # --- Initialize Parsers --- #
        
            

        # --- Prepare Prompt --- #
        # Re-enable refinement logic
        if current_retry > 0 and previous_plan_pydantic is not None and last_error is not None:
            logger.info("Refining previous plan due to errors.")
            prompt_template = PLANNER_REFINE_PROMPT_TEMPLATE
            # Try to serialize Pydantic model, fallback to raw string if needed
            try:
                previous_plan_str = previous_plan_pydantic.model_dump_json(indent=2)
            except AttributeError: # Handle cases where it might be a dict from earlier versions
                 previous_plan_str = json.dumps(previous_plan_pydantic, indent=2) if isinstance(previous_plan_pydantic, dict) else str(previous_plan_pydantic)
                 logger.warning(f"Could not use model_dump_json on previous plan, using json.dumps. Type was: {type(previous_plan_pydantic)}")
                 
            prompt_args = {
                "query": query,
                "tool_descriptions": filtered_tools_str,
                "previous_plan": previous_plan_str, # Pass the possibly raw previous plan
                "validation_errors": last_error
            }
        else:
            logger.info("Generating initial plan using filtered tools.")
            prompt_template = PLANNER_PROMPT_TEMPLATE
            prompt_args = {
                "query": query,
                "tool_descriptions": filtered_tools_str # USE FILTERED STRING
            }

        # --- Define Chain --- #
        plan_chain = prompt_template | llm

        # --- Invoke Chain --- #        
        logger.debug(f"Prompt arguments sent to LLM: {prompt_args}") # Log prompt args
        # Get raw response first for logging
        raw_llm_response = await (prompt_template | llm).ainvoke(prompt_args, config=node_config)
        raw_llm_response_content = raw_llm_response.content if hasattr(raw_llm_response, 'content') else str(raw_llm_response)
        logger.debug(f"Raw LLM response received:\n{raw_llm_response_content}")

        # Planner node now passes the raw YAML string to the parser node
        logger.info("Planner finished generating raw YAML plan. Proceeding to parser.")
        return {
            "raw_plan_output": raw_llm_response_content, # Store raw output for parser
            "workflow_status": "routing_complete",
            "error_message": None,
            "current_retry": current_retry,
            "next_node": "plan_parser" # Explicitly route to parser
        }

    except ValueError as e: # Keep ValueError catch? Maybe specific exception from LLM?
         # Catch parsing/validation errors specifically - MOVED TO PARSER
         logger.error(f"Planning failed due to LLM error or unexpected output structure before parsing: {e}", exc_info=True)
         error_msg = f"Planning failed: LLM Error - {e}"
         current_retry += 1
    except Exception as e:
        # Handle other exceptions (e.g., network issues during LLM call)
        # Log the raw response content if available
        if 'raw_llm_response_content' in locals():
            logger.error(f"Error during planning node attempt {current_retry + 1}. Raw LLM response was:\n{raw_llm_response_content}", exc_info=True)
        else:
            logger.error(f"Error during planning node attempt {current_retry + 1} (before LLM response was received or during prompt generation).", exc_info=True)
        
        error_msg = f"Planning failed: {e}"
        current_retry += 1
        if current_retry < max_retries:
            logger.warning(f"Retrying planning node (attempt {current_retry + 1}/{max_retries})...")
            return {
                "error_message": error_msg,
                "plan_pydantic": state.get('plan_pydantic'), # Keep previous valid plan if exists
                "current_retry": current_retry
            }
        else:
            logger.error(f"Max node retries ({max_retries}) reached. Failing workflow.")
            return {
                "error_message": error_msg,
                "workflow_status": "failed",
                "plan_pydantic": state.get('plan_pydantic'), # Keep previous valid plan if exists
                "plan": None,
                "evidence": {},
                "current_retry": current_retry,
                "next_node": END
            }