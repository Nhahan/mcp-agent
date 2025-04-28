from typing import List, Dict, Any, Optional, Tuple
import logging
from langsmith import traceable
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import BaseLanguageModel
import re # Import re for regex parsing
import json # Import json for parsing

# from ..state import ReWOOState, PlanOutput, PlanStepPydantic, ToolCallPydantic # This import might cause circular dependency if state imports planner node
# Let's try importing state directly
from agent.state import ReWOOState, PlanStep, PlanOutput # Adjusted import based on potential structure
# Define Pydantic models locally if needed or ensure state has them
# from pydantic import BaseModel, Field
# class ToolCallPydantic(BaseModel):
#     tool_name: str
#     arguments: Dict[str, Any]
# class PlanStepPydantic(BaseModel):
#     thought: str
#     tool_call: Optional[ToolCallPydantic] = None
#     expected_outcome: str

# Removed import from deleted planner
# from ..planner import _format_plan_to_string # Keep formatting helper for refinement
from ..prompts.plan_prompts import PLANNER_PROMPT_TEMPLATE, PLANNER_REFINE_PROMPT_TEMPLATE
# Import PlanValidator locally if needed or adjust path
# from ..validation import PlanValidator # Needs to exist or be moved
# --- Placeholder for PlanValidator --- #
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
            # Add more checks (tool_call format, etc.) if needed
        return not errors, errors
# --- End Placeholder for PlanValidator --- #

from langgraph.graph import END

# --- Logging Setup --- #
logger = logging.getLogger(__name__)
# --- End Logging Setup --- #

# --- Helper function moved from deleted agent/planner.py --- #
def _format_plan_to_string(plan_steps: List[Tuple[str, str, str, str]]) -> str:
    """Formats the plan steps (list of tuples) into a numbered string for the prompt."""
    formatted = ""
    for i, step_tuple in enumerate(plan_steps):
        reasoning, evidence_var, tool_name, tool_input = step_tuple
        formatted += f"Plan: {reasoning.strip()}\n"
        formatted += f"{evidence_var} = {tool_name}[{tool_input}]\n\n" # Keep original format
    return formatted.strip()
# --- End Helper function --- #

# Regex to find the JSON block within the LLM response (still needed)
JSON_BLOCK_REGEX = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL) # Use non-greedy match for content

@traceable(name="Planning Node")
async def planning_node(state: ReWOOState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Generates the initial plan based on the user query using standard LLM invocation.
    Parses the standard JSON structure expected from the updated prompt.
    Receives llm and tools_str via config['configurable'].
    Sets workflow_status to 'routing_complete' on success, 'failed' on error.
    """
    logger.info(f"--- Entering Planning Node (Attempt {state.get('current_retry', 0) + 1}) ---")
    max_retries = state.get('max_retries', 1)
    retries = state.get('current_retry', 0)

    if retries >= max_retries:
        logger.error(f"Max retries ({max_retries}) reached for planning.")
        return {"workflow_status": "failed", "error_message": f"Planning failed after {max_retries} retries.", "current_retry": retries, "next_node": END}

    logger.info(f"Generating plan for query: '{state['original_query']}'")
    logger.debug(f"Planning Node received state keys: {list(state.keys())}")

    configurable = config.get("configurable", {})
    llm = configurable.get("llm")
    tools_str = configurable.get("tools_str")
    if not llm or not tools_str:
        logger.error(f"LLM or tools_str missing in config['configurable']: {configurable}")
        return {"error_message": "LLM or tools_str missing in config", "workflow_status": "failed", "next_node": END}

    query = state["original_query"]
    last_error_list = None
    last_parsed_plan_attempt = state.get('plan')
    raw_llm_response = "" # Initialize to store raw response for logging

    while retries < max_retries:
        logger.info(f"Planning attempt {retries + 1}/{max_retries}")
        parsed_plan_tuples: List[Tuple[str, str, str, str]] = []
        plan_steps_json = [] # Store the raw parsed steps from JSON
        errors = []

        try:
            # --- Prepare Prompt ---
            # Refine prompt might need adjustment if we pass structured JSON instead of formatted string
            if retries > 0 and last_parsed_plan_attempt is not None and last_error_list is not None:
                logger.info("Refining previous plan due to errors.")
                prompt_template = PLANNER_REFINE_PROMPT_TEMPLATE
                formatted_last_plan = _format_plan_to_string(last_parsed_plan_attempt)
                prompt_args = {
                    "query": query,
                    "tool_descriptions": tools_str,
                    "previous_plan": formatted_last_plan,
                    "validation_errors": "\n".join([f"- {err}" for err in last_error_list])
                }
            else:
                logger.info("Generating initial plan using updated JSON prompt.")
                prompt_template = PLANNER_PROMPT_TEMPLATE
                prompt_args = {
                    "query": query,
                    "tool_descriptions": tools_str,
                }

            # --- Invoke LLM ---
            plan_chain = prompt_template | llm
            logger.info("Invoking LLM for plan generation...")
            response_str: str = await plan_chain.ainvoke(prompt_args, config=config)
            raw_llm_response = response_str # Store raw response
            log_message = f"Received raw response from LLM:\n```\n{response_str}\n```"
            logger.info(log_message)

            # --- Parse Standard JSON Response ---
            stripped_response = response_str.strip()
            json_match = JSON_BLOCK_REGEX.search(stripped_response)
            if not json_match:
                logger.error("Could not find ```json ... ``` block in stripped LLM response.")
                logger.debug(f"Stripped response was:\n```\n{stripped_response}\n```")
                raise ValueError("LLM response did not contain a ```json ... ``` block after stripping whitespace.")

            json_string = json_match.group(1).strip()
            try:
                plan_data = json.loads(json_string)
                if not isinstance(plan_data, dict) or "steps" not in plan_data or not isinstance(plan_data["steps"], list):
                     logger.error("Parsed JSON is not a dictionary or missing 'steps' list.")
                     raise ValueError("Parsed JSON is not a dictionary or missing 'steps' list.")
                plan_steps_json = plan_data["steps"] # Store the raw parsed steps
                logger.info(f"Successfully parsed standard JSON. Found {len(plan_steps_json)} steps.")

            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to parse standard JSON from LLM response: {json_err}")
                logger.debug(f"JSON string causing parsing error:\n```json\n{json_string}\n```")
                raise ValueError(f"Standard JSON parsing failed: {json_err}") from json_err

            # --- Extract Data and Check for Tool Calls ---
            if not plan_steps_json:
                 # This case means JSON was valid but steps list was empty
                 logger.warning("Parsed JSON contains an empty 'steps' list.")
                 errors.append("Plan is empty.") 
                 # Treat as error and retry
                 last_error_list = errors
                 last_parsed_plan_attempt = [] # No steps to refine
                 retries += 1
                 continue # Go to next iteration of the while loop

            # Iterate through steps to populate parsed_plan_tuples *only* for tool calls
            for i, step_json in enumerate(plan_steps_json):
                if not isinstance(step_json, dict):
                    logger.warning(f"Step {i+1} is not a dictionary, skipping: {step_json}")
                    errors.append(f"Step {i+1} is not a valid dictionary.")
                    continue # Skip this invalid step

                reasoning = step_json.get("plan")
                tool_call_data = step_json.get("tool_call")

                # Only process steps with actual tool calls for parsed_plan_tuples
                if tool_call_data is not None:
                    if not isinstance(tool_call_data, dict):
                        logger.warning(f"Step {i+1}: 'tool_call' is not a dictionary or null: {tool_call_data}")
                        errors.append(f"Step {i+1}: Invalid 'tool_call' format.")
                        continue # Skip this invalid step

                    evidence_var = tool_call_data.get("evidence_variable")
                    tool_name = tool_call_data.get("tool_name")
                    tool_arguments = tool_call_data.get("arguments")

                    # Validate tool call structure
                    valid_tool_call = True
                    if not evidence_var or not isinstance(evidence_var, str) or not evidence_var.startswith("#E"):
                        logger.warning(f"Step {i+1}: Missing or invalid 'evidence_variable': {evidence_var}")
                        errors.append(f"Step {i+1}: Missing/invalid 'evidence_variable'.")
                        valid_tool_call = False
                    if not tool_name or not isinstance(tool_name, str):
                        logger.warning(f"Step {i+1}: Missing or invalid 'tool_name': {tool_name}")
                        errors.append(f"Step {i+1}: Missing/invalid 'tool_name'.")
                        valid_tool_call = False
                    if tool_arguments is None or not isinstance(tool_arguments, dict):
                        if tool_arguments is not None:
                            logger.warning(f"Step {i+1}: 'arguments' is not a dictionary: {tool_arguments}")
                            errors.append(f"Step {i+1}: Invalid 'arguments' format (must be a JSON object/dict).")
                            valid_tool_call = False
                        else:
                            logger.debug(f"Step {i+1}: 'arguments' key missing in tool_call, assuming empty {{}}.")
                            tool_arguments = {}
                    
                    if valid_tool_call:
                        tool_input_str = json.dumps(tool_arguments)
                        if reasoning: # Ensure reasoning exists
                            parsed_plan_tuples.append((reasoning, evidence_var, tool_name, tool_input_str))
                        else:
                            logger.warning(f"Step {i+1} has a valid tool call but missing reasoning. Skipping tuple creation.")
                            errors.append(f"Step {i+1} missing reasoning ('plan' key).")
                    
                else:
                    logger.info(f"Step {i+1}: No tool call found (tool_call is null).")
                    # Check if reasoning exists even for null tool calls
                    if not reasoning or not isinstance(reasoning, str):
                         logger.warning(f"Step {i+1} (no tool call) missing reasoning ('plan' key).")
                         errors.append(f"Step {i+1} missing reasoning ('plan' key).")

            # --- Decide Next Step Based on Parsing Results ---
            if not errors:
                # Parsing successful, no structural errors found in steps
                if parsed_plan_tuples:
                    # Plan has tool calls, proceed to tool execution
                    logger.info(f"Plan successfully parsed with {len(parsed_plan_tuples)} tool-call steps. Proceeding to tool selection.")
                    return {
                        "plan": parsed_plan_tuples,
                        "current_step_index": 0,
                        "evidence": {}, # Reset evidence
                        "workflow_status": "routing_complete",
                        "error_message": None,
                        "current_retry": retries,
                        "next_node": "tool_selector"
                    }
                else:
                    # Plan has NO tool calls, but JSON was valid and steps existed.
                    # Go directly to final answer generation.
                    logger.info("Plan successfully parsed but contains no tool calls. Proceeding directly to final answer generation.")
                    # We need to pass the reasoning from the steps to the final answer node.
                    # Let's store the thoughts in the state? Or perhaps just the original query?
                    # For now, just route to final answer. The final answer node might need access to the original query.
                    # Store the raw parsed steps for the final answer node
                    return {
                        "plan": [], # Empty list as no tool calls
                        "all_parsed_steps": plan_steps_json, # Store all parsed JSON steps here
                        "current_step_index": 0, # Or set to -1? Does final answer need index?
                        "evidence": {}, # No evidence gathered
                        "workflow_status": "routing_complete",
                        "error_message": None,
                        "current_retry": retries,
                        "next_node": "generate_final_answer" # Skip tool execution
                    }
            else:
                # Errors encountered during step validation/extraction
                last_error_list = errors
                logger.warning(f"Plan parsing failed or incomplete due to errors in steps: {last_error_list}")
                last_parsed_plan_attempt = parsed_plan_tuples # Store potentially partially correct tuples for refine prompt
                retries += 1

        except Exception as e:
            # Handle JSON parsing errors or other unexpected errors
            logger.error(f"Error during planning attempt {retries + 1}", exc_info=True)
            error_context = f"Error: {e}. Raw LLM Response:\n```\n{raw_llm_response}\n```"
            error_type = "Parsing Error" if isinstance(e, (ValueError, json.JSONDecodeError)) else "Unexpected Planning Error"
            last_error_list = [f"{error_type}: {e}"]
            # If JSON parsing failed completely, plan_steps_json might not be defined
            last_parsed_plan_attempt = None # Can't use partially parsed tuples if JSON itself failed
            retries += 1

    # --- Max Retries Reached --- 
    error_msg = f"Planning failed after {max_retries} attempts. Last errors: {last_error_list}"
    logger.error(f"{error_msg}. Raw LLM Response from last attempt:\n```\n{raw_llm_response}\n```")
    return {
        "error_message": error_msg,
        "workflow_status": "failed",
        "plan": last_parsed_plan_attempt if last_parsed_plan_attempt is not None else None,
        "evidence": {},
        "current_retry": retries,
        "next_node": END
    } 