from typing import Dict, Any
import logging
import json

from ..state import ReWOOState
from ..prompts.answer_prompts import FINAL_ANSWER_PROMPT_TEMPLATE
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END

logger = logging.getLogger(__name__)


async def final_answer_node(state: ReWOOState, node_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates the final answer based on the original query and collected evidence.
    If evidence is empty, it uses the plan steps as context.
    Sets workflow_status to 'finished' on success, 'failed' on error.
    """
    logger.info("--- Starting Final Answer Node ---")

    # Extract LLM from node_config
    configurable = node_config.get("configurable", {})
    llm = configurable.get("llm")
    if not llm:
        logger.error("LLM not found in config['configurable'] for final_answer_node.")
        return {
            "final_answer": None,
            "error_message": "Final Answer Node Error: LLM not configured.",
            "workflow_status": "failed",
            "evidence": state.get("evidence", {}), # Preserve evidence
            "next_node": END
        }

    # Check for errors from previous steps before proceeding
    if state.get("workflow_status") == "failed":
        logger.warning("Skipping final answer generation due to previous failure.")
        # Return existing state, ensuring final_answer is None and evidence is preserved
        # Ensure evidence is returned as the correct type (dict)
        return {**state, "final_answer": None, "evidence": state.get("evidence", {})}

    original_query = state["original_query"]
    evidence_dict = state.get("evidence", {}) # Get evidence as dictionary
    # plan_steps = state.get("plan", []) # Get the plan steps (List[ParsedPlanStep]) - Not directly used for context
    max_retries = state.get("max_retries", 1)
    current_retries = 0
    last_error = None

    # Format context for the prompt: Use evidence if available, otherwise use plan_pydantic
    if evidence_dict:
        formatted_context = "\n".join([
            f"- Evidence {key}: {json.dumps(value) if isinstance(value, (dict, list)) else str(value)}"
            for key, value in evidence_dict.items()
        ])
        logger.debug(f"Using collected evidence for prompt:\n{formatted_context}")
    elif state.get("plan_pydantic") and state["plan_pydantic"].steps: # Check plan_pydantic
        plan_output = state["plan_pydantic"]
        # Format plan steps from Pydantic model if evidence is empty
        formatted_context = "Based on the generated plan (no tools were executed):\n"
        plan_steps_context = [f"- Step {i+1}: {step.plan}" for i, step in enumerate(plan_output.steps)]

        if plan_steps_context:
            formatted_context += "\n".join(plan_steps_context)
            logger.debug(f"Using plan steps from plan_pydantic for prompt (no evidence collected):\n{formatted_context}")
        else:
             # Should not happen if plan_pydantic.steps is not empty, but as safeguard
            formatted_context = "No evidence was collected and the generated plan steps were empty."
            logger.warning("Plan steps found in plan_pydantic, but they seem empty.")
    else:
        # Handle case where both evidence and plan_pydantic are empty/missing
        formatted_context = "No evidence was collected and no plan was generated or parsed."
        logger.warning("Neither evidence nor plan_pydantic are available for final answer generation.")


    # Use the prompt template
    prompt = FINAL_ANSWER_PROMPT_TEMPLATE.format(
        original_query=original_query,
        collected_evidence=formatted_context # Use the unified context
    )

    while current_retries <= max_retries:
        logger.info(f"Final Answer generation attempt {current_retries + 1}/{max_retries + 1}")
        try:
            # Invoke LLM
            response = await llm.ainvoke(prompt)
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