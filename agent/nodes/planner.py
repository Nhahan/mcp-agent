import logging
import json
from typing import Dict, Any
from langsmith import traceable
from ..state import ReWOOState
from ..prompts.plan_prompts import PLANNER_PROMPT_TEMPLATE, PLANNER_REFINE_PROMPT_TEMPLATE
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

    logger.debug(f"Planner using the following tool descriptions:\n---\n{filtered_tools_str}\n---")

    query = state["original_query"]
    last_error = state.get('error_message')
    previous_plan_pydantic = state.get('plan_pydantic')

    # Check if LLM is None before proceeding
    if not llm:
        logger.error(f"LLM (base or planner) missing in config['configurable']: {configurable}")
        return {"error_message": "LLM missing in config", "workflow_status": "failed", "current_retry": current_retry, "next_node": END}

    try:
        if current_retry > 0 and previous_plan_pydantic is not None and last_error is not None:
            logger.info("Refining previous plan due to errors.")
            prompt_template = PLANNER_REFINE_PROMPT_TEMPLATE
            # Try to serialize Pydantic model, fallback to raw string if needed
            try:
                previous_plan_str = previous_plan_pydantic.model_dump_json(indent=2)
            except AttributeError: # Handle cases where it might be a dict from earlier versions
                 previous_plan_str = json.dumps(previous_plan_pydantic, indent=2) if isinstance(previous_plan_pydantic, dict) else str(previous_plan_pydantic)
                 logger.warning(f"Could not use model_dump_json on previous plan, using json.dumps. Type was: {type(previous_plan_pydantic)}")
                 
            # For refinement, error_history is derived from last_error
            error_history_for_prompt = f"Previous Error: {last_error}"
            prompt_args = {
                "query": query,
                "tool_descriptions": filtered_tools_str,
                "raw_plan_output": previous_plan_str, # PLANNER_REFINE_PROMPT_TEMPLATE expects raw_plan_output
                "error_message": last_error, # And error_message for the specific error
                "error_history": error_history_for_prompt # And a general error_history
            }
        else:
            logger.info("Generating initial plan using filtered tools.")
            prompt_template = PLANNER_PROMPT_TEMPLATE
            # For initial plan, error_history is typically empty or not applicable
            error_history_for_prompt = state.get("error_history_str", "N/A") # Get from state or use default
            prompt_args = {
                "query": query,
                "tool_descriptions": filtered_tools_str, # USE FILTERED STRING
                "error_history": error_history_for_prompt # Add error_history for initial prompt too
            }

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