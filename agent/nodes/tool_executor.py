# agent/nodes/tool_executor.py
from typing import Dict, Any, List, Optional
import logging

from ..state import ReWOOState
from langchain_core.tools import BaseTool
from langgraph.graph import END
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)

# Helper function to find a tool by name from the list
def find_tool_by_name(tools: List[BaseTool], name: str) -> Optional[BaseTool]:
    for tool in tools:
        if tool.name == name:
            return tool
    return None

async def tool_execution_node(state: ReWOOState, node_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes the selected tool with the prepared input using BaseTool objects.
    Retrieves tool name, input, and evidence var from state.
    Retrieves the list of BaseTool objects from the config.
    Stores output/error in the evidence dictionary.
    Sets workflow_status to route to 'evidence_processor' on success, 'failed' on error.
    """
    logger.info("--- Starting Tool Execution Node ---")
    current_tool_call_info = state.get("current_tool_call")
    prepared_input = state.get("prepared_tool_input")
    current_evidence_dict = state.get("evidence", {})
    current_step_index = state.get("current_step_index", 0)
    step_number = current_step_index + 1

    # Retrieve filtered tools list from state
    filtered_tools_list = state.get("filtered_tools", [])
    if not filtered_tools_list:
        # Correctly handle potential None for current_tool_call_info in log message
        tool_name_for_log = current_tool_call_info.get('tool_name') if current_tool_call_info else 'unknown'
        logger.warning(f"No tools found in state['filtered_tools'] at step {step_number}. Cannot execute tool '{tool_name_for_log}'.")
        # Fallback to getting tools from config if state is empty (should not happen ideally)
        # Use the passed node_config dictionary for fallback
        configurable = node_config.get("configurable", {}) 
        filtered_tools_list = configurable.get("tools", [])
        if not filtered_tools_list:
             logger.error(f"No tools found in state OR config at step {step_number}. Failing.")
             return {
                 "error_message": f"Internal error: No tools available for execution at step {step_number}.",
                 "workflow_status": "failed",
                 "next_node": END
             }
        logger.warning("Using full tool list from config as fallback.")

    # --- Defensive Checks --- #
    if not current_tool_call_info:
        logger.error(f"Missing current_tool_call info at step {step_number}.")
        return {
            "error_message": f"Internal error: Missing tool call info at step {step_number}.",
            "workflow_status": "failed",
            "next_node": END
        }

    tool_name = current_tool_call_info.get("tool_name")
    evidence_var = current_tool_call_info.get("evidence_var")

    if not tool_name or not evidence_var:
        logger.error(f"Tool name ('{tool_name}') or evidence variable ('{evidence_var}') missing in current_tool_call at step {step_number}.")
        return {
            "error_message": f"Internal error: Missing tool name or evidence var at step {step_number}.",
            "workflow_status": "failed",
            "evidence": current_evidence_dict,
            "next_node": END
        }

    # Find the actual tool object from the FILTERED list
    tool_to_execute = find_tool_by_name(filtered_tools_list, tool_name)

    if not tool_to_execute:
        logger.error(f"Tool '{tool_name}' specified in plan (Step {step_number}) not found in the filtered tools list: {[t.name for t in filtered_tools_list]}")
        # --- Modify Error Handling to FAIL immediately --- #
        error_msg = f"Tool '{tool_name}' not found in relevant tools at step {step_number}. Filtered list: {[t.name for t in filtered_tools_list]}"
        logger.error(f"Failing workflow because invalid or non-filtered tool '{tool_name}' was specified.")
        return {
            "error_message": error_msg,
            "workflow_status": "failed",
            "evidence": current_evidence_dict,
            "next_node": END # Fail immediately
        }
        # --- End Modification --- #
    # --- End Defensive Checks --- #

    output_evidence: Any = None
    next_status = "routing_complete"
    next_node = "evidence_processor"
    error_message = None

    try:
        logger.info(f"(Step {step_number}) Executing tool: {tool_name} with input: {prepared_input}")

        # Invoke the tool with the prepared input (which might be {})
        # Pass the node_config dictionary to ainvoke
        tool_result = await tool_to_execute.ainvoke(prepared_input, config=node_config)

        # --- Check if the result itself is an error message --- #
        if isinstance(tool_result, str) and (
            tool_result.startswith("Error:") or 
            tool_result.startswith("Invalid arguments") or
            "fail" in tool_result.lower() # Add other common error indicators if needed
        ):
            error_message = f"Tool {tool_name} returned an error string: {tool_result}"
            logger.error(error_message)
            output_evidence = error_message # Store error string as evidence
            next_status = "failed"
            next_node = END # Halt on tool execution errors
            # Skip the successful execution logs below
        else:
            # --- REMOVE specific sequentialthinking handling ---
            # Store the raw tool result object (could be dict, str, list, etc.)
            logger.debug(f"(Step {step_number}) Raw tool result type: {type(tool_result)}, value: {str(tool_result)[:500]}...") # Log type and part of value
            output_evidence = tool_result # Store the raw object, not str(tool_result)
            # ---------------------------------------------------
            
            logger.info(f"(Step {step_number}) Tool '{tool_name}' executed successfully.")
            # Keep success status and node
            next_status = "routing_complete"
            next_node = "evidence_processor"
            error_message = None

    except Exception as e:
        error_message = f"Exception during tool execution for {tool_name} (Step {step_number}): {e}"
        logger.error(error_message, exc_info=True)
        output_evidence = error_message # Store exception string as evidence
        next_status = "failed"
        next_node = END # Halt on tool execution errors

    # Store the raw evidence object (or error message string) in the dictionary
    logger.debug(f"Before update - current_evidence_dict type: {type(current_evidence_dict)}, value: {current_evidence_dict}")
    logger.debug(f"Before update - evidence_var type: {type(evidence_var)}, value: {evidence_var}")
    logger.debug(f"Before update - output_evidence type: {type(output_evidence)}, value: {str(output_evidence)[:100]}...")
    current_evidence_dict[evidence_var] = output_evidence

    logger.debug(f"After update - current_evidence_dict: {current_evidence_dict}") # Add after update log

    next_step_index = current_step_index + 1

    # Restore original return logic
    return {
        "evidence": current_evidence_dict,
        "workflow_status": next_status,
        "error_message": error_message,
        "next_node": next_node,
        "current_step_index": next_step_index
    } 