# agent/nodes/plan_validator.py
import logging
from typing import Dict, Any, List, Optional
from langgraph.graph import END
# Remove RunnableConfig import if no longer needed
# from langchain_core.runnables import RunnableConfig

from ..state import ReWOOState, PlanOutputPydantic

logger = logging.getLogger(__name__)

# FIX: Change the second argument from config: RunnableConfig to node_config: Dict
async def plan_validator_node(state: ReWOOState, node_config: Dict) -> Dict[str, Any]:
    """
    Validates the parsed plan (Pydantic model) from the plan_parser node.
    Checks if all tool names used in the plan exist in the list of available tools
    retrieved from the state ('filtered_tools').
    Routes back to the planner for refinement if invalid tools are found,
    otherwise routes to the next appropriate node (tool_selector or final_answer).
    """
    logger.info("--- Entering Plan Validator Node ---")
    parsed_plan: Optional[PlanOutputPydantic] = state.get("plan_pydantic")
    # Retrieve max_retries from node_config if needed, or keep getting from state
    max_retries = state.get('max_retries', 1) # Or potentially node_config['configurable'].get(...)
    current_retry = state.get('current_retry', 0)

    # Retrieve the list of FILTERED tool names from state
    # This part remains unchanged as it correctly uses the state
    filtered_tools = state.get("filtered_tools", [])
    if not filtered_tools:
        logger.warning("Plan Validator Warning: No filtered tools found in state['filtered_tools']. Cannot validate tool names.")
        available_tool_names = []
    else:
        available_tool_names = [tool.name for tool in filtered_tools]

    # --- Basic Checks ---
    if not parsed_plan:
        logger.error("Plan Validator Error: No parsed plan (plan_pydantic) found in state.")
        return {
            "error_message": "Plan Validator Error: No parsed plan found.",
            "workflow_status": "failed",
            "next_node": END # Ensure next_node is set for error cases handled by should_continue
        }

    invalid_tool_found = False
    first_invalid_tool_name = ""

    logger.debug(f"Validating plan with {len(parsed_plan.steps)} steps against FILTERED tools: {available_tool_names}")
    for i, step in enumerate(parsed_plan.steps):
        if step.tool_call:
            tool_name_to_check = step.tool_call.tool_name
            logger.debug(f"Step {i+1}: Checking tool '{tool_name_to_check}' against filtered list...")
            # Handle the case where available_tool_names is empty (no tools filtered/available)
            if not available_tool_names:
                 logger.error(f"Invalid tool found in plan step {i+1}: '{tool_name_to_check}'. No tools are available in the filtered list.")
                 invalid_tool_found = True
                 first_invalid_tool_name = tool_name_to_check
                 break
            elif tool_name_to_check not in available_tool_names:
                logger.error(f"Invalid tool found in plan step {i+1}: '{tool_name_to_check}'. Not in the filtered list of relevant tools.")
                invalid_tool_found = True
                first_invalid_tool_name = tool_name_to_check
                break # Stop validation on first invalid tool

    if invalid_tool_found:
        error_msg = f"Invalid tool '{first_invalid_tool_name}' found in plan (not in relevant filtered list). Filtered tools: {available_tool_names}"
        retries = current_retry + 1 # Increment retry based on planner/parser attempts
        if retries < max_retries:
            logger.warning(f"Routing back to planner for refinement due to invalid tool (Attempt {retries}/{max_retries}).")
            # Return state needed for planner refinement
            return {
                "error_message": error_msg,
                "current_retry": retries,
                 # Keep existing plan_pydantic for the refine prompt
                "plan_pydantic": parsed_plan,
                "raw_plan_output": state.get("raw_plan_output"), # Keep raw output if needed by planner refine logic
                "next_node": "planner"
            }
        else:
            logger.error(f"Max retries ({max_retries}) reached after invalid tool validation. Failing workflow.")
            return {
                "error_message": error_msg + f" (Failed after {max_retries} attempts).",
                "workflow_status": "failed", # Signal failure
                "current_retry": retries,
                "next_node": END # Explicitly route to END for failure
            }
    else:
        logger.info("Plan validation successful. All tool names are valid.")
        next_node = "generate_final_answer" # Default if no tool calls
        # Check if there are any steps with a tool_call in the Pydantic plan
        has_tool_calls = any(step.tool_call for step in parsed_plan.steps)

        if has_tool_calls:
            logger.info("Plan contains tool calls. Routing to tool_selector.")
            next_node = "tool_selector"
        else:
            logger.info("Plan contains no tool calls (only thought/final answer). Routing to generate_final_answer.")

        return {
            "workflow_status": "routing_complete", # Indicate successful validation and routing decision
            "error_message": None, # Clear any previous error
            "current_retry": 0, # Reset retries on successful validation
            "next_node": next_node
            # Keep plan_pydantic and plan in state for subsequent nodes
        } 