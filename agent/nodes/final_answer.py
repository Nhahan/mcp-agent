from typing import Dict, Any
import logging
import json

from ..state import ReWOOState
from ..prompts.answer_prompts import FINAL_ANSWER_PROMPT_TEMPLATE
from langchain_core.language_models import BaseLanguageModel
from langgraph.graph import END

# --- Logging Setup --- #
logger = logging.getLogger(__name__)
# --- End Logging Setup --- #


async def final_answer_node(state: ReWOOState, llm: BaseLanguageModel) -> Dict[str, Any]:
    """
    Generates the final answer based on the original query and collected evidence.
    If evidence is empty, it uses the plan steps as context.
    Sets workflow_status to 'finished' on success, 'failed' on error.
    """
    logger.info("--- Starting Final Answer Node ---")
    # Check for errors from previous steps before proceeding
    if state.get("workflow_status") == "failed":
        logger.warning("Skipping final answer generation due to previous failure.")
        # Return existing state, ensuring final_answer is None and evidence is preserved
        # Ensure evidence is returned as the correct type (dict)
        return {**state, "final_answer": None, "evidence": state.get("evidence", {})}

    original_query = state["original_query"]
    evidence_dict = state.get("evidence", {}) # Get evidence as dictionary
    plan_steps = state.get("plan", []) # Get the plan steps (List[ParsedPlanStep])
    max_retries = state.get("max_retries", 1)
    current_retries = 0
    last_error = None

    # Format context for the prompt: Use evidence if available, otherwise use plan
    if evidence_dict:
        formatted_context = "\n".join([
            f"- Evidence {key}: {json.dumps(value) if isinstance(value, (dict, list)) else str(value)}"
            for key, value in evidence_dict.items()
        ])
        logger.debug(f"Using collected evidence for prompt:\n{formatted_context}")
    elif state.get("all_parsed_steps"): # Check 'all_parsed_steps' instead of 'plan'
        all_steps = state["all_parsed_steps"]
        # Format plan steps if evidence is empty
        formatted_context = "Based on the generated plan (no tools were executed):\n"
        valid_steps_context = []
        for i, step_dict in enumerate(all_steps):
            plan_text = step_dict.get("plan") # Extract 'plan' text from each dict
            if plan_text:
                valid_steps_context.append(f"- Step {i+1}: {plan_text}")
            else:
                logger.warning(f"Step {i+1} in all_parsed_steps is missing 'plan' key.")
        
        if valid_steps_context:
            formatted_context += "\n".join(valid_steps_context)
            logger.debug(f"Using plan steps from all_parsed_steps for prompt (no evidence collected):\n{formatted_context}")
        else:
             # Handle case where all_parsed_steps exists but contains no valid plan text
            formatted_context = "No evidence was collected and the generated plan steps were invalid or empty."
            logger.warning("Plan steps found in all_parsed_steps, but they lack valid 'plan' content.")
    else:
        # Handle case where both evidence and plan are empty
        formatted_context = "No evidence was collected and no plan was generated."
        logger.warning("Neither evidence nor plan steps are available for final answer generation.")


    # Use the prompt template
    prompt = FINAL_ANSWER_PROMPT_TEMPLATE.format(
        original_query=original_query,
        collected_evidence=formatted_context # Use the unified context
    )

    while current_retries <= max_retries:
        logger.info(f"Final Answer generation attempt {current_retries + 1}/{max_retries + 1}")
        try:
            # Invoke LLM
            response = await llm.ainvoke(prompt) # Use ainvoke for consistency
            response_content = response.content if hasattr(response, 'content') else str(response)
            logger.debug(f"Raw LLM Response:\n{response_content}")
            final_answer = response_content.strip()
            if not final_answer:
                raise ValueError("LLM returned an empty final answer.")

            logger.info(f"Generated Final Answer: {final_answer}")
            # Return success state, preserving evidence dictionary
            return {
                "final_answer": final_answer,
                "workflow_status": "finished",
                "error_message": None,
                "evidence": evidence_dict, # Preserve evidence dictionary
                "next_node": END
            }
        except Exception as e:
            logger.error(f"Error during final answer generation attempt {current_retries + 1}", exc_info=True)
            last_error = e
            current_retries += 1

    # If loop finishes without success
    logger.error(f"Max retries ({max_retries + 1}) reached for final answer generation.")
    error_msg = f"Final answer generation failed after {max_retries + 1} attempts: {last_error}"
    # Return failure state, preserving evidence dictionary
    return {
        "final_answer": None,
        "error_message": error_msg,
        "workflow_status": "failed",
        "evidence": evidence_dict, # Preserve evidence dictionary
        "next_node": END
    } 