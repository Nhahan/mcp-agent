# agent/graph.py
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from functools import partial
import asyncio
import json
import re # Import re for parsing in final_answer_node
import logging # Import logging
from contextlib import asynccontextmanager # Import asynccontextmanager

# Import state definition and component functions
from .state import ReWOOState, PlanStep, Evidence, ToolResult, ToolCall
from .planner import generate_plan, parse_plan, _format_plan_to_string, _format_tool_descriptions
from .solver import generate_final_answer
# Import MCP client from adapter library
from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.prompts.plan_prompts import PLANNER_PROMPT_TEMPLATE, PLANNER_REFINE_PROMPT_TEMPLATE
from agent.prompts.answer_prompts import FINAL_ANSWER_PROMPT_TEMPLATE
from agent.validation import PlanValidator
from agent.tools.registry import ToolRegistry

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel

# --- Logging Setup --- #
# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)
# --- End Logging Setup --- #

# Helper function to find a tool by name from the list
def find_tool_by_name(tools: List[BaseTool], name: str) -> Optional[BaseTool]:
    for tool in tools:
        if tool.name == name:
            return tool
    return None

# --- LangGraph Nodes ---

async def planning_node(state: ReWOOState, llm: BaseLanguageModel, tools: List[BaseTool]) -> Dict[str, Any]:
    """
    Generates the initial plan based on the user query using planner functions.
    Sets workflow_status to 'route_to_selector' on success, 'failed' on error.
    """
    logger.info("--- Starting Planning Node ---")
    query = state["original_query"]
    max_retries = state.get("max_retries", 1)
    current_retries = 0
    last_error_list = None
    last_plan = None

    validator = PlanValidator() # Assume no tool names needed for basic validation here
    tool_descriptions = _format_tool_descriptions(tools) # Format tool descriptions

    while current_retries <= max_retries:
        logger.info(f"Planning attempt {current_retries + 1}/{max_retries + 1}")
        try:
            # Correct the keys for the prompt template
            prompt_args = {
                "query": query, # Use 'query' key
                "tool_descriptions": tool_descriptions # Add tool descriptions
            }
            if current_retries > 0 and last_plan is not None and last_error_list is not None:
                logger.info("Refining previous plan due to validation errors.")
                prompt_template = PLANNER_REFINE_PROMPT_TEMPLATE
                prompt_args['previous_plan'] = _format_plan_to_string(last_plan)
                prompt_args['validation_errors'] = "\n".join([f"- {err}" for err in last_error_list])
            else:
                logger.info("Generating initial plan.")
                prompt_template = PLANNER_PROMPT_TEMPLATE

            # Chain the prompt and LLM
            plan_chain = prompt_template | llm
            response = await plan_chain.ainvoke(prompt_args)
            response_content = response.content if hasattr(response, 'content') else str(response)
            logger.debug(f"Raw LLM Response:\n{response_content}")

            # Use the imported parse_plan function
            current_plan = parse_plan(response_content)
            logger.info(f"Parsed Plan: {current_plan}")

            if not current_plan:
                 # If parsing returns empty list, even if LLM returned content, consider it an error
                 if response_content.strip():
                     raise ValueError("LLM returned content but parsing resulted in an empty plan.")
                 else:
                     raise ValueError("LLM returned empty content.")

            # Validate the plan
            is_valid, errors = validator.validate(current_plan)

            if is_valid:
                logger.info("Plan validation successful.")
                return {
                    "plan": current_plan,
                    "current_step_index": 0,
                    "evidence": [],
                    "workflow_status": "route_to_selector", # Explicit routing instruction
                    "error_message": None
                }
            else:
                # Extract error messages for refinement prompt
                last_error_list = [f"Step {idx if idx >=0 else 'N/A'}: {desc[1]}" for idx, desc in enumerate(errors)]
                logger.warning(f"Plan validation failed: {last_error_list}")
                last_plan = current_plan
                current_retries += 1

        except Exception as e:
            logger.error(f"Error during planning attempt {current_retries + 1}", exc_info=True)
            last_error_list = [f"Unexpected planning error: {e}"]
            # Keep the last plan if it exists, otherwise it remains None
            last_plan = current_plan if current_plan else last_plan
            current_retries += 1

    logger.error(f"Max retries ({max_retries + 1}) reached for planning.")
    # Explicitly set workflow status to failed
    return {
        "error_message": f"Planning failed after {max_retries + 1} attempts. Last errors: {last_error_list}",
        "workflow_status": "failed",
        "plan": last_plan or [],
        "evidence": []
    }

async def tool_selection_node(state: ReWOOState) -> Dict[str, Any]:
    """
    Selects the tool for the current step based on the plan.
    Routes to final answer generation if the plan is complete or an error occurred.
    Updates workflow_status to 'tool_input_preparation', 'final_answer', or 'tool_selection' (loop).
    """
    logger.info("--- Starting Tool Selection Node ---")
    # Check for prior errors first - THIS CHECK IS REDUNDANT if planner routes correctly
    # if state.get("error_message") and state.get("workflow_status") == "failed":
    #     logger.warning("Previous node failed. Routing to END.")
    #     return {"workflow_status": "failed"} # Let routing handle failed status

    plan = state["plan"]
    current_step_index = state["current_step_index"]

    # Handle empty plan (should have been caught by planner routing, but defensive check)
    if not plan:
        logger.error("Tool selector called with empty plan. Routing to END.")
        return {"workflow_status": "failed", "error_message": "Tool selector called with empty plan."} 

    if current_step_index >= len(plan):
        logger.info("Plan finished. Moving to final answer generation.")
        return {"workflow_status": "final_answer"} # Signal to route to final answer

    current_step = plan[current_step_index]
    tool_call_data = current_step.get("tool_call") # This is now a dict or None
    step_number = current_step_index + 1

    if not tool_call_data or not isinstance(tool_call_data, dict):
        logger.info(f"No tool call required for step {step_number}. Considering step complete (thought only).")
        # Collect thought as evidence and move to the next step
        thought_evidence = f"Step {step_number} Thought: {current_step.get('thought', '(No thought provided)')}"
        logger.debug(f"Adding thought as evidence: {thought_evidence}")
        current_evidence = state.get("evidence", [])
        return {
            "current_step_index": current_step_index + 1,
            "evidence": current_evidence + [thought_evidence],
            "tool_name": None, # Ensure tool_name is None for thought-only step
            "workflow_status": "tool_selection" # Loop back to select for the *next* step
        }

    # Extract tool name
    tool_name = tool_call_data.get("tool_name")
    if not tool_name:
        logger.error(f"Tool name missing in tool_call data for step {step_number}: {tool_call_data}")
        # Treat as failure for this step
        return {
            "error_message": f"Tool name missing for step {step_number}.",
            "workflow_status": "failed",
            "current_step_index": current_step_index + 1 # Increment index to avoid infinite loop
        }

    logger.info(f"Selected tool: '{tool_name}' for step {step_number}")
    # Set tool_name and signal to prepare input
    return {"tool_name": tool_name, "workflow_status": "tool_input_preparation"}

async def tool_input_preparation_node(state: ReWOOState) -> Dict[str, Any]:
    """
    Prepares the input arguments for the selected tool based on the current plan step.
    Handles substituting evidence placeholders like #E1, #E2 etc.
    Sets workflow_status to 'tool_execution' on success, 'failed' on error.
    """
    logger.info("--- Starting Tool Input Preparation Node ---")
    plan = state["plan"]
    current_step_index = state["current_step_index"]
    tool_name = state.get("tool_name")
    evidence_list = state.get("evidence", []) # Get collected evidence
    step_number = current_step_index + 1

    # --- Defensive Checks --- #
    if tool_name is None:
         logger.error(f"Tool name is None in state at step {step_number}. Aborting preparation.")
         return {"workflow_status": "failed", "error_message": f"Tool name missing at step {step_number}"}
    if current_step_index >= len(plan):
        logger.error(f"Index out of bounds ({current_step_index}) at step {step_number}. Aborting preparation.")
        return {"workflow_status": "failed", "error_message": "Tool input preparation called after plan finished."}
    # --- End Defensive Checks --- #

    current_step = plan[current_step_index]
    tool_call_data = current_step.get("tool_call") # dict or None

    # Defensive check (should be caught by selection node)
    if not tool_call_data or not isinstance(tool_call_data, dict) or tool_call_data.get("tool_name") != tool_name:
        logger.error(f"Tool call data mismatch or missing for step {step_number}. Expected tool: '{tool_name}', got: {tool_call_data}")
        return {"workflow_status": "failed", "error_message": f"Tool call data mismatch for step {step_number}."}

    # Get arguments from the parsed tool_call dictionary
    raw_args = tool_call_data.get("arguments", {})
    if not isinstance(raw_args, dict):
        logger.warning(f"Arguments for step {step_number} are not a dict: {raw_args}. Attempting to use anyway.")
        # If it's not a dict, how to substitute? Treat as error or single arg?
        # Let's treat as error for now, planner should ensure args are dict.
        return {"workflow_status": "failed", "error_message": f"Invalid argument format (not a dict) for step {step_number}."}

    try:
        # Substitute evidence placeholders (#E1, #E2, ...)
        logger.debug(f"Raw parsed input args for step {step_number}: {raw_args}")
        substituted_input = _substitute_evidence_in_args(raw_args, evidence_list, step_number)
        logger.info(f"Prepared input for '{tool_name}' (Step {step_number}): {substituted_input}")

        return {"tool_input": substituted_input, "workflow_status": "tool_execution"}
    except Exception as e:
        logger.error(f"Error substituting arguments for tool call at step {step_number}", exc_info=True)
        return {"workflow_status": "failed", "error_message": f"Failed to substitute arguments for step {step_number}: {e}"}

def _substitute_evidence_in_args(args: Dict[str, Any], evidence_list: List[Any], step_number_for_log: int) -> Dict[str, Any]:
    """ Substitutes #E<n> placeholders in args values with actual evidence. """
    # Create a map from placeholder (e.g., #E1) to the corresponding evidence
    # Assuming evidence_list stores the direct output/error string for simplicity now
    evidence_map = {f"#E{i+1}": evidence for i, evidence in enumerate(evidence_list)}
    logger.debug(f"(Step {step_number_for_log}) Evidence map for substitution: {evidence_map}")

    def recursive_substitute(value: Any) -> Any:
        if isinstance(value, str):
            # Use regex to find all #E<n> occurrences
            def replace_match(match):
                placeholder = match.group(0)
                if placeholder in evidence_map:
                    # Replace with the actual evidence
                    # Return the evidence directly (could be string, dict, list...)
                    # Be careful if downstream tool expects only strings!
                    logger.debug(f"(Step {step_number_for_log}) Substituting placeholder '{placeholder}'")
                    return evidence_map[placeholder]
                else:
                    # Allow missing evidence, return placeholder or raise error?
                    logger.warning(f"(Step {step_number_for_log}) Evidence placeholder '{placeholder}' not found. Keeping placeholder.")
                    return placeholder # Keep placeholder if not found

            # Substitute all occurrences in the string
            # We need to handle the case where the evidence itself is not a string
            # This simple regex substitution assumes we want string replacement
            # If evidence can be other types, this needs more complex handling
            new_value_str, num_subs = re.subn(r"#E\d+", replace_match, value)

            # If the *entire* string was a placeholder and replaced with non-string, return the object
            if num_subs == 1 and value in evidence_map and not isinstance(evidence_map[value], str):
                 logger.debug(f"(Step {step_number_for_log}) Placeholder '{value}' replaced with non-string object.")
                 return evidence_map[value]
            else:
                 return new_value_str

        elif isinstance(value, dict):
            return {k: recursive_substitute(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [recursive_substitute(item) for item in value]
        else:
            return value # Handle numbers, booleans, etc.

    return recursive_substitute(args)

async def tool_execution_node(state: ReWOOState, tool_registry: ToolRegistry) -> Dict[str, Any]:
    """
    Executes the selected tool with the prepared input and stores the output as evidence.
    Increments the step index.
    Sets workflow_status to 'tool_selection' on success, 'failed' on error.
    """
    logger.info("--- Starting Tool Execution Node ---")
    tool_name = state.get("tool_name")
    tool_input = state.get("tool_input", {})
    current_step_index = state.get("current_step_index", 0)
    current_evidence = state.get("evidence", [])
    step_number = current_step_index + 1

    if not tool_name:
        logger.error("Tool name missing.")
        return {
            "error_message": "Tool name not found in state for execution.",
            "workflow_status": "failed",
            "evidence": current_evidence + ["Error: Tool name missing."],
            "current_step_index": current_step_index + 1 # Increment index
        }

    output_evidence = None
    next_status = "tool_selection" # Default next step is selecting tool for next step

    try:
        tool_callable = tool_registry.get_tool(tool_name)
        if not tool_callable:
            error_msg = f"Error: Tool '{tool_name}' not found in registry."
            logger.error(f"(Step {step_number}) {error_msg}")
            output_evidence = error_msg # Store error as evidence
            next_status = "failed" # Mark workflow as failed
        else:
            logger.info(f"(Step {step_number}) Executing tool: {tool_name} with input: {tool_input}")
            # Execute the tool - assuming synchronous for now
            tool_output = tool_callable(**tool_input) # Pass input as keyword arguments
            logger.info(f"(Step {step_number}) Raw Tool '{tool_name}' output: {str(tool_output)[:500]}...")
            output_evidence = tool_output # Store the actual output

    except Exception as e:
        error_msg = f"Error executing tool '{tool_name}' (Step {step_number}): {str(e)}"
        logger.error(error_msg, exc_info=True)
        output_evidence = error_msg # Store error as evidence
        next_status = "failed" # Mark workflow as failed

    # Increment step index for the next iteration
    next_step_index = current_step_index + 1

    # Append the evidence (could be successful output or error message)
    updated_evidence = current_evidence + [output_evidence]
    logger.debug(f"(Step {step_number}) Evidence collected: {str(output_evidence)[:100]}...")

    return {
        "current_step_index": next_step_index,
        "evidence": updated_evidence,
        "workflow_status": next_status, # Either 'tool_selection' or 'failed'
        "error_message": state.get("error_message") if next_status == "failed" else None # Keep/set error if failed
    }

async def final_answer_node(state: ReWOOState, llm: BaseLanguageModel) -> Dict[str, Any]:
    """
    Generates the final answer based on the original query and collected evidence.
    Sets workflow_status to 'finished' on success, 'failed' on error.
    """
    logger.info("--- Starting Final Answer Node ---")
    # Check for errors from previous steps before proceeding
    if state.get("workflow_status") == "failed":
        logger.warning("Skipping final answer generation due to previous failure.")
        # Return existing state, ensuring final_answer is None and evidence is preserved
        return {**state, "final_answer": None, "evidence": state.get("evidence", [])}

    original_query = state["original_query"]
    evidence_list = state.get("evidence", [])
    max_retries = state.get("max_retries", 1)
    current_retries = 0
    last_error = None

    # Format evidence for the prompt
    formatted_evidence = "\n".join([
        f"- Evidence from Step {i+1}: {json.dumps(ev) if isinstance(ev, (dict, list)) else str(ev)}"
        for i, ev in enumerate(evidence_list)
    ])
    if not formatted_evidence:
        formatted_evidence = "No evidence was collected."
    logger.debug(f"Evidence for prompt:\n{formatted_evidence}")

    # Use the prompt template
    prompt = FINAL_ANSWER_PROMPT_TEMPLATE.format(
        original_query=original_query,
        collected_evidence=formatted_evidence
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
            # *** Ensure evidence is preserved on success ***
            return {
                "final_answer": final_answer,
                "workflow_status": "finished",
                "error_message": None,
                "evidence": state.get("evidence", [])
            }
        except Exception as e:
            logger.error(f"Error during final answer generation attempt {current_retries + 1}", exc_info=True)
            last_error = e
            current_retries += 1

    # If loop finishes without success
    logger.error(f"Max retries ({max_retries + 1}) reached for final answer generation.")
    error_msg = f"Final answer generation failed after {max_retries + 1} attempts: {last_error}"
    # *** Preserve evidence on failure too ***
    return {
        "final_answer": None,
        "error_message": error_msg,
        "workflow_status": "failed",
        "evidence": state.get("evidence", [])
    }

# --- Conditional Edge Logic (Routing based on workflow_status) ---

def route_workflow(state: ReWOOState) -> str:
    """Determines the next node based on the workflow_status set by the previous node."""
    status = state.get("workflow_status")
    logger.info(f"--- Routing based on status: '{status}' ---")

    if status == "failed":
        logger.info("Routing -> END (Workflow Failed)")
        return "__end__"
    elif status == "route_to_selector": # Check for the new explicit status
        logger.info("Routing -> tool_selector")
        return "tool_selector" # Return the node name directly
    elif status == "tool_input_preparation":
        logger.info("Routing -> tool_input_preparer")
        return "tool_input_preparer"
    elif status == "tool_execution":
        logger.info("Routing -> tool_executor")
        return "tool_executor"
    elif status == "final_answer": # Status set by tool_selector when plan is done
        logger.info("Routing -> final_answer_node")
        return "final_answer_node"
    elif status == "finished": # Status set by final_answer_node on success
        logger.info("Routing -> END (Workflow Finished)")
        return "__end__"
    # Handle the loop back for thought-only steps from tool_selector
    elif status == "tool_selection": # This status should now only be set by tool_selector for looping
        logger.info("Routing -> tool_selector (Looping back for next step/thought)")
        return "tool_selector"
    else:
        # Default/fallback: if status is unexpected or None, end the graph
        logger.warning(f"Routing -> END (Unknown or missing status: '{status}')")
        return "__end__"

# --- Graph Construction --- #

def build_rewoo_graph(llm: BaseLanguageModel, tools: List[BaseTool], tool_registry: ToolRegistry) -> StateGraph:
    """ Creates and configures the LangGraph StateGraph for the ReWOO agent. """
    workflow = StateGraph(ReWOOState)

    # Bind components to nodes
    bound_planning_node = partial(planning_node, llm=llm, tools=tools)
    bound_tool_executor_node = partial(tool_execution_node, tool_registry=tool_registry)
    bound_final_answer_node = partial(final_answer_node, llm=llm)

    # Add nodes (use descriptive names)
    workflow.add_node("planner", bound_planning_node)
    workflow.add_node("tool_selector", tool_selection_node)
    workflow.add_node("tool_input_preparer", tool_input_preparation_node)
    workflow.add_node("tool_executor", bound_tool_executor_node)
    workflow.add_node("final_answer_node", bound_final_answer_node) # Corrected node name

    # Set entry point
    workflow.set_entry_point("planner")

    # Define edges using the route_workflow function
    # Each node (except entry) should decide the *next* intended status

    # Planner routes based on its success/failure
    workflow.add_conditional_edges(
        "planner",
        route_workflow, # Uses status set by planner node ('route_to_selector' or 'failed')
        {
            "tool_selector": "tool_selector", # If route_workflow returns 'tool_selector'
            "__end__": END                  # If route_workflow returns '__end__'
        }
    )

    # Tool selector routes based on its decision
    workflow.add_conditional_edges(
        "tool_selector",
        route_workflow, # Uses status set by tool_selector ('tool_input_preparation', 'final_answer', 'tool_selection', 'failed')
        {
            "tool_input_preparer": "tool_input_preparer",
            "final_answer_node": "final_answer_node",
            "tool_selector": "tool_selector", # For looping back
            "__end__": END
        }
    )

    # Tool input preparer routes based on its success/failure
    workflow.add_conditional_edges(
        "tool_input_preparer",
        route_workflow, # Uses status set by preparer ('tool_execution' or 'failed')
        {
            "tool_executor": "tool_executor", # Corrected: Map status 'tool_execution' to node 'tool_executor'
            "__end__": END                 # Map status 'failed' to END
        }
    )

    # Tool executor routes based on its success/failure
    workflow.add_conditional_edges(
        "tool_executor",
        route_workflow, # Uses status set by executor ('tool_selection' or 'failed')
        {
            "tool_selector": "tool_selector", # Corrected: Map status/return value 'tool_selector' to node 'tool_selector'
            "__end__": END                 # Map status 'failed' to END
        }
    )

    # Final answer node routes based on its success/failure
    workflow.add_conditional_edges(
        "final_answer_node",
        route_workflow, # Uses status set by final answer node ('finished' or 'failed')
        {
            "__end__": END
        }
    )

    # Compile the graph
    app = workflow.compile()
    logger.info("ReWOO graph compiled successfully with updated routing.")
    return app

# --- Example Usage (run_test) remains largely the same --- #
async def run_test():
    # Mock LLM (as before)
    from langchain_community.llms.fake import FakeListLLM
    # Define fake plan string matching the expected output format from the prompt
    # Using Step X: format for the corrected parser
    fake_plan_str = """
Step 1:
Thought: Search for LangGraph.
Tool Call: search(query='LangGraph')
Expected Outcome: Link to LangGraph documentation.

Step 2:
Thought: Summarize the main concepts using the search result from Step 1.
Tool Call: summarize(content=#E1)
Expected Outcome: A brief summary.
"""

    fake_search_result = "LangGraph is a library..."
    fake_summary = "LangGraph helps build complex apps..."
    fake_final_answer = "LangGraph is a library for building stateful LLM applications..."

    # Correct responses sequence for FakeListLLM: Plan, Final Answer
    fake_llm = FakeListLLM(responses=[
        fake_plan_str, # Planner response string
        fake_final_answer # Final Answer response string
    ])

    # Mock Tool Registry and register mock tools
    tool_registry = ToolRegistry()
    def mock_search(query: str):
        logger.info(f"MOCK SEARCH CALLED with query: {query}")
        if query == 'LangGraph': return fake_search_result
        return "No results found."
    def mock_summarize(content: str):
        logger.info(f"MOCK SUMMARIZE CALLED with content: {str(content)[:50]}...")
        if content == fake_search_result: return fake_summary
        return "Could not summarize."
    tool_registry.register_tool("search", mock_search, "Mock search tool")
    tool_registry.register_tool("summarize", mock_summarize, "Mock summarize tool")

    # Mock BaseTool list for planner description formatting
    # Need BaseTool structure for _format_tool_descriptions
    class MockTool(BaseTool):
        name: str
        description: str
        def _run(self, *args, **kwargs): pass
        async def _arun(self, *args, **kwargs): pass

    mock_tools_for_planner = [
        MockTool(name="search", description="Mock search tool"),
        MockTool(name="summarize", description="Mock summarize tool")
    ]

    logger.info("\n--- Building ReWOO Graph (within test) ---")
    app = build_rewoo_graph(llm=fake_llm, tools=mock_tools_for_planner, tool_registry=tool_registry)

    logger.info("\n--- Invoking Graph ---")
    # Corrected initial state with original_query
    initial_state = ReWOOState(
        original_query="What is LangGraph and what is it used for?",
        plan=[],
        current_step_index=0,
        tool_name=None,
        tool_input=None,
        # tool_output=None, # Removed as output is stored in evidence
        evidence=[],
        final_answer=None,
        error_message=None,
        max_retries=1,
        # current_retry=0, # Removed as retry logic is within nodes now
        workflow_status='planning' # Start with planning status
    )
    final_state = None
    try:
        # Use astream to capture intermediate states
        async for event in app.astream(initial_state, {"recursion_limit": 15}): # Increased limit slightly
             for key, value in event.items():
                 logger.info(f"\n--- State after Node '{key}' --- Workflow Status: '{value.get('workflow_status')}'")
                 # Log only key fields for brevity
                 log_state = {
                     k: v for k, v in value.items()
                     if k in ["workflow_status", "current_step_index", "tool_name", "final_answer", "error_message", "evidence"]
                 }
                 # Truncate evidence for logging
                 log_state["evidence"] = [str(e)[:100] + "..." if isinstance(e, str) and len(e) > 100 else e for e in log_state.get("evidence", [])]
                 logger.info(json.dumps(log_state, indent=2, default=str))
                 final_state = value # Keep track of the latest state
        logger.info("\n--- Graph Execution Complete ---")
    except Exception as e:
        logger.error("Graph execution failed", exc_info=True)

    logger.info("\n--- Final State ---")
    if final_state:
         # More detailed final state logging
         final_log_state = {k: v for k, v in final_state.items() if k != 'plan'} # Exclude potentially long plan
         final_log_state["plan_step_count"] = len(final_state.get("plan", []))
         logger.info(json.dumps(final_log_state, indent=2, default=str))

         # Activate and adjust assertions
         assert final_state.get("final_answer") == fake_final_answer, f"Expected final answer '{fake_final_answer}', got '{final_state.get('final_answer')}'"
         assert final_state.get("workflow_status") == "finished", f"Expected status 'finished', got '{final_state.get('workflow_status')}'"
         assert final_state.get("error_message") is None, f"Expected no error, got '{final_state.get('error_message')}'"
         # Check collected evidence
         evidence = final_state.get("evidence", [])
         assert len(evidence) == 2, f"Expected 2 pieces of evidence, got {len(evidence)}"
         assert evidence[0] == fake_search_result, f"Evidence 1 mismatch. Got: {evidence[0]}"
         assert evidence[1] == fake_summary, f"Evidence 2 mismatch. Got: {evidence[1]}"
         logger.info("\nAssertions Passed!")
    else:
        logger.error("Graph did not return a final state.")


# Remove the old run_graph_example which depends on the context manager logic
# Keep run_test as the primary test function for now
if __name__ == "__main__":
     asyncio.run(run_test()) 