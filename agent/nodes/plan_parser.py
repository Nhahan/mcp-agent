import json
import yaml
from typing import Dict, Any, List, Tuple, Optional
import logging
import re
from langgraph.graph import END
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from agent.state import ReWOOState
from ..state import PlanOutputPydantic
from ..prompts.parser_prompts import PLAN_CORRECTOR_PROMPT

logger = logging.getLogger(__name__)

ParsedPlanStep = Tuple[str, str, str, str] # (plan_step_thought, evidence_var, tool_name, tool_input_json_str)

def _extract_yaml_block(text: str) -> Optional[str]:
    """Extracts the first YAML code block or the raw text if no block found."""
    logger.debug("Attempting to extract YAML block...")
    # Regex to find YAML block ```yaml ... ``` or ``` ... ```
    yaml_block_match = re.search(r"```(?:yaml)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if yaml_block_match:
        extracted = yaml_block_match.group(1).strip()
        logger.debug(f"Extracted YAML block:\n{extracted}")
        return extracted
    else:
        # If no block markers, assume the entire text is the intended YAML (strip leading/trailing whitespace)
        # Remove potential preamble/epilogue often added by LLMs
        lines = text.strip().split('\n')
        # Find the start of the YAML structure (likely 'steps:')
        start_index = -1
        for i, line in enumerate(lines):
            if line.strip().startswith("steps:"):
                start_index = i
                break
        if start_index != -1:
             potential_yaml = "\n".join(lines[start_index:]).strip()
             logger.debug(f"No YAML block found, assuming raw text starting from 'steps:' is YAML:\n{potential_yaml}")
             return potential_yaml
        else:
             logger.warning("No YAML block markers found and 'steps:' not found in raw output. Returning raw stripped text.")
             return text.strip()

async def _try_llm_correction(
    raw_output: str,
    config: RunnableConfig,
    node_config: Dict,
    error_context: Optional[str] = None,
    tool_name: Optional[str] = None, # Keep for potential future use in prompt? Or remove? Keep for now.
    tool_schema: Optional[Dict] = None # Keep for potential future use in prompt? Or remove? Keep for now.
) -> Optional[str]:
    """Attempts to correct the raw output using an LLM call, optionally using tool schema."""
    log_message = "Initial YAML parsing failed or structure invalid."
    if error_context:
        log_message += f" Specific Error Context: {error_context}"
    if tool_name and tool_schema:
        log_message += f" Providing schema context for tool '{tool_name}' for correction." # Modified log
    logger.warning(log_message + " Attempting LLM correction...")

    try:
        llm = node_config['configurable'].get('llm')
        if not llm:
            logger.error("LLM instance not found in node_config for correction.")
            return None

        # Prepare prompt input, providing defaults for schema if not available
        prompt_input = {
            "raw_planner_output": raw_output,
            "error_context": error_context if error_context else "N/A",
            # Pass tool name/schema even if prompt doesn't explicitly use them for detailed correction yet
            # Prompt currently focuses on making arguments a valid YAML mapping
            "tool_name": tool_name if tool_name else "N/A",
            "tool_schema": json.dumps(tool_schema, indent=2) if tool_schema else "{}"
        }

        correction_prompt = PLAN_CORRECTOR_PROMPT.format(**prompt_input)
        logger.debug(f"Correction prompt:\n{correction_prompt}")

        correction_response = await llm.ainvoke(correction_prompt, config=config)

        if hasattr(correction_response, 'content'):
            corrected_yaml_str = correction_response.content
        elif isinstance(correction_response, str):
            corrected_yaml_str = correction_response
        else:
             logger.error(f"Unexpected LLM response type for correction: {type(correction_response)}")
             return None

        logger.info(f"LLM correction attempt response:\n{corrected_yaml_str}")
        final_corrected_yaml = _extract_yaml_block(corrected_yaml_str)
        if final_corrected_yaml:
             logger.info("Successfully extracted YAML from LLM correction response.")
             return final_corrected_yaml
        else:
             logger.warning("Could not extract YAML block from LLM correction response.")
             return None

    except Exception as e:
        logger.error(f"Error during LLM correction attempt: {e}", exc_info=True)
        return None

async def plan_parser_node(state: ReWOOState, config: RunnableConfig, node_config: Dict) -> Dict[str, Any]:
    """
    Parses the raw plan output from the planner node.
    Attempts extraction, YAML parsing, and LLM correction for structure/validation errors.
    Validates the result using Pydantic models (with extra='forbid').
    Populates 'plan_pydantic', 'plan', and 'all_parsed_steps' in the state.
    """
    logger.info("--- Entering Plan Parser Node ---")
    raw_plan_output: Optional[str] = state.get("raw_plan_output")
    # Tool map only needed temporarily for error analysis context
    # available_tools: List[BaseTool] = node_config.get('configurable', {}).get('tools', [])
    # tool_map: Dict[str, BaseTool] = {tool.name: tool for tool in available_tools}
    max_retries = state.get('max_retries', 1)
    current_retry = state.get('current_retry', 0)

    if not raw_plan_output:
        logger.error("Plan Parser Error: No raw plan output found in state.")
        return {"error_message": "Planner did not return output.", "workflow_status": "failed", "next_node": END}

    parsed_yaml: Optional[Dict] = None
    plan_pydantic: Optional[PlanOutputPydantic] = None
    yaml_string_to_parse: Optional[str] = None
    error_message: Optional[str] = None # Combined parsing/validation error message
    correction_attempted = False

    # --- Loop for Parsing, Validation, and LLM Correction ---
    while True:
        parsing_error = None
        pydantic_error = None
        error_tool_name = None # Still useful for LLM context
        error_tool_schema = None # Still useful for LLM context

        if yaml_string_to_parse is None:
            yaml_string_to_parse = _extract_yaml_block(raw_plan_output)

        if not yaml_string_to_parse:
            parsing_error = "Could not extract YAML block or content."
        else:
            try:
                parsed_yaml = yaml.safe_load(yaml_string_to_parse)
                if not isinstance(parsed_yaml, dict) or "steps" not in parsed_yaml or not isinstance(parsed_yaml.get("steps"), list):
                    raise yaml.YAMLError("Parsed YAML does not have the expected 'steps' list structure.")
                # Final Pydantic validation (now stricter with extra='forbid')
                try:
                    plan_pydantic = PlanOutputPydantic(**parsed_yaml)
                    logger.info("YAML parsed and validated successfully against Pydantic model.")
                    error_message = None # Clear any previous error
                    break # Success! Exit loop.
                except ValidationError as e:
                    pydantic_error = f"Pydantic validation failed (extra='forbid' active): {e}" # More specific error message
                    logger.error(f"{pydantic_error}.")
                    # Analyze error for context (keep this part)
                    try:
                        # Need tool_map here temporarily just for error analysis
                        # Re-import BaseTool here if needed for the map
                        from langchain_core.tools import BaseTool
                        tool_map_for_analysis: Dict[str, BaseTool] = {tool.name: tool for tool in node_config.get('configurable', {}).get('tools', [])}
                        first_error = e.errors()[0]
                        if len(first_error['loc']) >= 3 and first_error['loc'][0] == 'steps' and first_error['loc'][2] == 'tool_call':
                           step_index = int(first_error['loc'][1])
                           if parsed_yaml and 'steps' in parsed_yaml and step_index < len(parsed_yaml['steps']):
                                step_data = parsed_yaml['steps'][step_index]
                                if isinstance(step_data.get('tool_call'), dict):
                                    error_tool_name = step_data['tool_call'].get('tool_name')
                                    if error_tool_name and error_tool_name in tool_map_for_analysis:
                                        schema = getattr(tool_map_for_analysis[error_tool_name], 'args', None)
                                        if hasattr(schema, 'model_json_schema'): error_tool_schema = schema.model_json_schema()
                                        elif isinstance(schema, dict): error_tool_schema = schema
                                        else: error_tool_schema = None
                                        if not error_tool_schema: error_tool_name = None
                    except Exception: pass
                    # Fall through to LLM correction
            except yaml.YAMLError as e:
                parsing_error = f"YAML parsing failed: {e}"
                # Fall through to LLM correction

        # --- LLM Correction Trigger ---
        if parsing_error or pydantic_error:
            error_message = pydantic_error if pydantic_error else parsing_error # Store the latest error
            if not correction_attempted:
                logger.warning("Attempting LLM correction due to parsing or validation error.")
                string_to_correct = yaml_string_to_parse if yaml_string_to_parse else raw_plan_output
                corrected_yaml_str = await _try_llm_correction(string_to_correct, config, node_config, error_message, error_tool_name, error_tool_schema)
                correction_attempted = True
                if corrected_yaml_str:
                    yaml_string_to_parse = corrected_yaml_str
                    parsed_yaml = None # Force re-parsing
                    logger.info("Using LLM corrected YAML string for next attempt.")
                    continue # Retry parsing/validation
                else:
                    error_message = (error_message if error_message else "Unknown Error") + ". LLM correction also failed."
                    break # Exit loop, correction failed
            else:
                 error_message = (error_message if error_message else "Unknown Error") + " (after LLM correction attempt)."
                 break # Exit loop, correction already failed
        else:
             # Should only happen if Pydantic validation passed
             break # Exit loop

    # --- Handle Failure or Success ---
    if error_message: # If loop exited with an error
        retries = current_retry + 1
        if retries < max_retries:
             logger.warning(f"Plan parsing/validation failed. Routing back to planner (Attempt {retries}/{max_retries}). Error: {error_message}")
             return {"error_message": error_message, "raw_plan_output": raw_plan_output, "current_retry": retries, "next_node": "planner"}
        else:
             logger.error(f"Plan parsing/validation failed after {max_retries} attempts. Failing workflow. Error: {error_message}")
             return {"error_message": error_message + f" (Failed after {max_retries} attempts).", "workflow_status": "failed", "current_retry": retries, "next_node": END}

    # --- Success Case: Prepare output state ---
    if plan_pydantic: # Should be valid if loop succeeded
        plan_tuples: List[ParsedPlanStep] = []
        all_parsed_steps_list: List[Dict[str, Any]] = []
        try:
            for i, step in enumerate(plan_pydantic.steps):
                 step_dict = {"step_index": i + 1, "plan": step.plan, "tool_call": None, "status": "pending"}
                 if step.tool_call:
                      # Arguments are validated as dict by Pydantic
                      parsed_args = step.tool_call.arguments
                      args_json_str = json.dumps(parsed_args) # Serialize for tuple format
                      plan_tuples.append((step.plan, step.tool_call.evidence_variable, step.tool_call.tool_name, args_json_str))
                      
                      # Prepare the tool_call dictionary for all_parsed_steps
                      tool_call_dict_for_steps = {
                           "evidence_variable": step.tool_call.evidence_variable,
                           "tool_name": step.tool_call.tool_name,
                           "arguments": parsed_args # Keep as dict
                      }
                      # Add evidence_input_key if it exists in the Pydantic model (Planner should ideally set this)
                      if step.tool_call.evidence_input_key:
                          tool_call_dict_for_steps["evidence_input_key"] = step.tool_call.evidence_input_key
                      
                      step_dict["tool_call"] = tool_call_dict_for_steps
                      
                 all_parsed_steps_list.append(step_dict)

            logger.info("Successfully prepared plan_pydantic, plan_tuples, and all_parsed_steps.") # Revert log message
            has_tool_calls = any(step.tool_call for step in plan_pydantic.steps)
            logger.info(f"Plan contains {len(plan_tuples)} executable tool calls.")
            # Route to the plan validator node (which checks tool names etc.)
            return {
                "plan_pydantic": plan_pydantic,
                "plan": plan_tuples,
                "all_parsed_steps": all_parsed_steps_list,
                "current_step_index": 0,
                "error_message": None,
                "current_retry": 0,
                "next_node": "plan_validator" # <--- ROUTE TO PLAN VALIDATOR ON SUCCESS
            }
        except Exception as final_prep_e:
             logger.error(f"Error preparing final state after successful validation: {final_prep_e}", exc_info=True)
             return {"error_message": f"Internal error preparing state: {final_prep_e}", "workflow_status": "failed", "next_node": END}
    else:
        # Should be unreachable
        logger.error("Internal Logic Error: Reached end of plan_parser_node without valid plan_pydantic or error.")
        return {"error_message": "Internal parser logic error (final stage).", "workflow_status": "failed", "next_node": END}