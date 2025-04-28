from typing import List, Dict, Any, Optional, Tuple
import logging
from langsmith import traceable
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
import re
import json
from agent.state import ReWOOState, ParsedPlanStep, PlanOutputPydantic, PlanStepPydantic, ToolCallPydantic
from ..prompts.plan_prompts import PLANNER_PROMPT_TEMPLATE, PLANNER_REFINE_PROMPT_TEMPLATE

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
def _format_plan_to_string(plan_steps: List[ParsedPlanStep]) -> str:
    """Formats the plan steps (list of tuples) into a numbered string for the prompt."""
    formatted = ""
    for i, step_tuple in enumerate(plan_steps):
        reasoning, evidence_var, tool_name, tool_input_str = step_tuple
        formatted += f"Step {i+1}:\nPlan: {reasoning.strip()}\nTool Call: {evidence_var} = {tool_name}[{tool_input_str}]\n\n"
    return formatted.strip()
# --- End Helper function --- #

# Regex to find the JSON block within the LLM response (still needed)
JSON_BLOCK_REGEX = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL) # Use non-greedy match for content

@traceable(name="Planning Node")
async def planning_node(state: ReWOOState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Generates or refines a plan using the LLM and Pydantic/OutputFixing parsers.
    Sets workflow_status to 'routing_complete' on success, 'failed' on error.
    """
    logger.info(f"--- Entering Planning Node (Attempt {state.get('current_retry', 0) + 1}) ---")
    max_retries = state.get('max_retries', 1)
    llm_retries = 1
    retries = state.get('current_retry', 0)

    if retries >= max_retries:
        logger.error(f"Max retries ({max_retries}) reached for planning node.")
        return {"workflow_status": "failed", "error_message": f"Planning failed after {max_retries} node retries.", "current_retry": retries, "next_node": END}

    logger.info(f"Generating plan for query: '{state['original_query']}'")
    logger.debug(f"Planning Node received state keys: {list(state.keys())}")

    configurable = config.get("configurable", {})
    llm = configurable.get("llm")
    tools_str = configurable.get("tools_str")
    if not llm or not tools_str:
        logger.error(f"LLM or tools_str missing in config['configurable']: {configurable}")
        return {"error_message": "LLM or tools_str missing in config", "workflow_status": "failed", "next_node": END}

    query = state["original_query"]
    last_error = state.get('error_message')
    previous_plan_pydantic = state.get('plan_pydantic')
    raw_llm_response_for_log = ""

    try:
        # --- Initialize Parsers --- #
        pydantic_parser = PydanticOutputParser(pydantic_object=PlanOutputPydantic)
        output_parser = OutputFixingParser.from_llm(
            parser=pydantic_parser,
            llm=llm,
            max_retries=llm_retries
        )
        format_instructions = output_parser.get_format_instructions()

        # --- Prepare Prompt --- #
        if retries > 0 and previous_plan_pydantic is not None and last_error is not None:
            logger.info("Refining previous plan due to errors.")
            prompt_template = PLANNER_REFINE_PROMPT_TEMPLATE
            previous_plan_str = json.dumps(previous_plan_pydantic.dict(), indent=2)
            prompt_args = {
                "query": query,
                "tool_descriptions": tools_str,
                "previous_plan": previous_plan_str,
                "validation_errors": last_error,
                "format_instructions": format_instructions
            }
        else:
            logger.info("Generating initial plan using Pydantic prompt.")
            prompt_template = PLANNER_PROMPT_TEMPLATE
            prompt_args = {
                "query": query,
                "tool_descriptions": tools_str,
                "format_instructions": format_instructions
            }

        # --- Define Chain --- #
        plan_chain = prompt_template | llm | output_parser

        # --- Invoke Chain --- #
        logger.info("Invoking LLM chain with output parser...")
        parsed_plan_output: PlanOutputPydantic = await plan_chain.ainvoke(prompt_args, config=config)
        logger.info(f"Successfully parsed plan using PydanticOutputParser. Found {len(parsed_plan_output.steps)} steps.")

        # --- Extract Tool Call Tuples --- #
        parsed_plan_tuples: List[ParsedPlanStep] = []
        for step in parsed_plan_output.steps:
            if step.tool_call:
                tool_input_str = json.dumps(step.tool_call.arguments)
                parsed_plan_tuples.append((
                    step.plan,
                    step.tool_call.evidence_variable,
                    step.tool_call.tool_name,
                    tool_input_str
                ))
            else:
                logger.info(f"Step: '{step.plan}' - No tool call.")

        # --- Decide Next Node --- #
        if parsed_plan_tuples:
            logger.info(f"Plan contains {len(parsed_plan_tuples)} tool calls. Proceeding to tool execution.")
            return {
                "plan_pydantic": parsed_plan_output,
                "plan": parsed_plan_tuples,
                "current_step_index": 0,
                "evidence": {},
                "workflow_status": "routing_complete",
                "error_message": None,
                "current_retry": retries,
                "next_node": "tool_selector"
            }
        else:
            logger.info("Plan contains no tool calls. Proceeding directly to final answer generation.")
            return {
                "plan_pydantic": parsed_plan_output,
                "plan": [],
                "current_step_index": 0,
                "evidence": {},
                "workflow_status": "routing_complete",
                "error_message": None,
                "current_retry": retries,
                "next_node": "generate_final_answer"
            }

    except Exception as e:
        # Handle exceptions potentially raised by the parser or LLM
        logger.error(f"Error during planning node attempt {retries + 1}", exc_info=True)
        error_msg = f"Planning failed: {e}"
        retries += 1
        if retries < max_retries:
            logger.warning(f"Retrying planning node (attempt {retries + 1}/{max_retries})...")
            return {
                "error_message": error_msg,
                "plan_pydantic": previous_plan_pydantic,
                "current_retry": retries
            }
        else:
            logger.error(f"Max node retries ({max_retries}) reached. Failing workflow.")
            return {
                "error_message": error_msg,
                "workflow_status": "failed",
                "plan_pydantic": previous_plan_pydantic,
                "plan": None,
                "evidence": {},
                "current_retry": retries,
                "next_node": END
            } 