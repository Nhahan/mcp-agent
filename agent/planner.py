# agent/planner.py
import re
import json
from typing import List, Optional
import logging

from langchain_core.language_models import BaseLanguageModel
# from langchain_core.prompts import PromptTemplate # Use the one from prompts module
from langchain_core.tools import BaseTool

from .state import PlanStep, ToolCall
# Import the specific prompt template
from .prompts.plan_prompts import PLANNER_PROMPT_TEMPLATE, PLANNER_REFINE_PROMPT_TEMPLATE
# Import the validator
from .validation import PlanValidator, ValidationIssue

# Configure logging at the module level
logger = logging.getLogger(__name__)

# Remove old template definition
# PLANNER_TEMPLATE = """..."""

# Updated Regex to capture tool_name and arguments (JSON or string) more reliably
# Captures: #E<num> = tool_name[ <arguments_json_or_string> ]
# Group 1: Evidence Num, Group 2: Tool Name, Group 3: Arguments String
TOOL_CALL_REGEX = re.compile(r"(\w+)\((.*)\)", re.DOTALL)
# Regex for Thought and Expected Outcome remain the same
THOUGHT_REGEX = re.compile(r"Thought:\s*(.*)", re.IGNORECASE)
EXPECTED_OUTCOME_REGEX = re.compile(r"Expected Outcome:\s*(.*)", re.IGNORECASE)


def _format_tool_descriptions(tools: List[BaseTool]) -> str:
    """ Formats the tool descriptions for the planner prompt. """
    descriptions = []
    for tool in tools:
        # Using name and description is usually sufficient for the planner
        # schema_desc = tool.args_schema.schema() if tool.args_schema else 'No arguments'
        # descriptions.append(f"- {tool.name}: {tool.description}\n  Args: {schema_desc}")
        descriptions.append(f"- {tool.name}: {tool.description}")
    return "\n".join(descriptions)

def _format_plan_to_string(plan: List[PlanStep]) -> str:
    """ Formats a plan list back into a string representation for the refinement prompt. """
    plan_str = ""
    for step in plan:
        plan_str += f"Thought: {step.get('thought', '')}\n"
        if step.get('tool_call'):
            tool_call = step.get('tool_call')
            # Assuming #E numbers match step_index + 1. Adjust if parser logic changes.
            evidence_num = step.get('step_index', -1) + 1
            # Basic formatting, might need more robust JSON string handling for args
            args_str = json.dumps(tool_call.get('arguments', {}))
            plan_str += f"Tool Call: #E{evidence_num} = {tool_call.get('tool_name')}[{args_str}]\n"
        if step.get('expected_outcome'):
            plan_str += f"Expected Outcome: {step.get('expected_outcome')}\n"
        plan_str += "\n" # Add blank line between steps
    return plan_str.strip()

def parse_plan(plan_str: str) -> List[PlanStep]:
    """Parses the LLM output string into a list of PlanStep dictionaries."""
    steps = []
    # Split the plan string into potential steps based on "Step X:"
    potential_steps = re.split(r"Step \d+:", plan_str)[1:] # Skip the part before the first "Step 1:"

    for i, step_text in enumerate(potential_steps):
        step_text = step_text.strip()
        if not step_text:
            continue

        thought = re.search(r"Thought: (.*?)\n", step_text, re.DOTALL)
        tool_call_match = re.search(r"Tool Call: (.*?)\n", step_text, re.DOTALL)
        expected_outcome = re.search(r"Expected Outcome: (.*?)$", step_text, re.DOTALL)

        thought_text = thought.group(1).strip() if thought else ""
        tool_call_str = tool_call_match.group(1).strip() if tool_call_match else ""
        expected_outcome_text = expected_outcome.group(1).strip() if expected_outcome else ""

        tool_name = None
        arguments = {}

        if tool_call_str:
            match = TOOL_CALL_REGEX.match(tool_call_str)
            if match:
                tool_name = match.group(1)
                args_str = match.group(2).strip()
                try:
                    # First, try parsing as JSON
                    arguments = json.loads(args_str)
                    if not isinstance(arguments, dict):
                        # Handle cases where JSON parsing results in a non-dict (e.g., just a string)
                        logger.warning(f"Parsed JSON arguments for tool '{tool_name}' is not a dict: {arguments}. Treating as single 'input' arg.")
                        arguments = {'input': args_str}
                except json.JSONDecodeError:
                    # If JSON parsing fails, try simple key=value parsing (for test cases)
                    try:
                        parsed_args = {}
                        # Regex to find key=value pairs, handling simple quotes
                        # Example: query='LangGraph', content=#E1
                        for arg_match in re.finditer(r"(\w+)\s*=\s*('([^']*)'|\"([^\"]*)\"|([^,]+))", args_str):
                            key = arg_match.group(1)
                            # Prioritize quoted values, then unquoted
                            value = arg_match.group(3) or arg_match.group(4) or arg_match.group(5)
                            parsed_args[key] = value.strip()
                        
                        if parsed_args:
                             arguments = parsed_args
                        else:
                             # If key=value parsing also fails, treat the whole string as a single 'input' argument
                            logger.warning(f"Could not parse arguments as key-value pairs for tool '{tool_name}'. Input: '{args_str}'. Treating as single 'input' arg.")
                            arguments = {'input': args_str}
                    except Exception as e:
                        logger.warning(f"Error during key-value argument parsing for tool '{tool_name}'. Input: '{args_str}'. Error: {e}. Treating as single 'input' arg.")
                        arguments = {'input': args_str}
                except Exception as e:
                    # Catch any other unexpected errors during parsing
                    logger.error(f"Unexpected error parsing arguments for tool '{tool_name}'. Input: '{args_str}'. Error: {e}. Treating as single 'input' arg.")
                    arguments = {'input': args_str}
            else:
                # If regex doesn't match tool_name(args) format, treat as thought
                 logger.warning(f"Tool call string '{tool_call_str}' did not match expected format. Treating step as thought-only.")
                 thought_text += f" (Tool Call Failed: {tool_call_str})" # Append failed call to thought
                 tool_call_str = "" # Clear tool call


        step_data = {
            "step_index": i,
            "thought": thought_text,
            "tool_call": {"tool_name": tool_name, "arguments": arguments} if tool_name else None,
            "expected_outcome": expected_outcome_text,
            "status": "pending" # Initial status
        }
        steps.append(step_data)

    if steps:
        logger.info(f"--- Parsed Plan Steps ({len(steps)} steps) ---")
        for i, step in enumerate(steps):
            logger.info(f"Step {i}:")
            logger.info(f"  Thought: {step['thought']}")
            logger.info(f"  Tool Call: {step['tool_call']}")
            logger.info(f"  Expected Outcome: {step['expected_outcome']}")
        logger.info("--------------------------")
    else:
        logger.warning("Plan parsing resulted in zero steps.")


    return steps


async def generate_plan(
    llm: BaseLanguageModel,
    tools: List[BaseTool],
    query: str,
    max_retries: int = 1 # Add max_retries for refinement loop later
) -> List[PlanStep]:
    """
    Generates and validates a plan using the LLM, attempting refinement if validation fails.

    Args:
        llm: The language model instance to use for planning.
        tools: A list of available Langchain tools.
        query: The user's input query.
        max_retries: Maximum attempts to refine the plan if validation fails.

    Returns:
        A list of PlanStep objects representing the validated (or best-effort) plan.
    """
    tool_descriptions = _format_tool_descriptions(tools)
    available_tool_names = [tool.name for tool in tools]
    validator = PlanValidator(available_tool_names)
    prompt_template = PLANNER_PROMPT_TEMPLATE # Start with the initial prompt

    current_plan: List[PlanStep] = []
    last_error: Optional[str] = None

    for attempt in range(max_retries + 1):
        print(f"\n--- Plan Generation Attempt {attempt + 1}/{max_retries + 1} ---")

        # Prepare prompt arguments
        generation_prompt_args = {
            "tool_descriptions": tool_descriptions,
            "query": query
        }

        if attempt > 0:
            # This is a retry attempt, use the refinement prompt and add context
            if last_error and current_plan:
                prompt_template = PLANNER_REFINE_PROMPT_TEMPLATE
                generation_prompt_args['previous_plan'] = _format_plan_to_string(current_plan)
                generation_prompt_args['validation_errors'] = last_error
                print("Attempting to refine plan based on validation errors...")
            else:
                # Cannot refine if there was no previous plan or error description
                print("Warning: Cannot refine plan without previous plan details or validation errors. Retrying with initial prompt.")
                prompt_template = PLANNER_PROMPT_TEMPLATE
                # Remove refinement-specific keys if they were accidentally added
                generation_prompt_args.pop('previous_plan', None)
                generation_prompt_args.pop('validation_errors', None)
        else:
             # First attempt uses the initial prompt
             prompt_template = PLANNER_PROMPT_TEMPLATE

        # Create the chain for this attempt
        planner_chain = prompt_template | llm

        try:
            llm_response = await planner_chain.ainvoke(generation_prompt_args)

            if hasattr(llm_response, 'content'):
                raw_plan_str = llm_response.content
            elif isinstance(llm_response, str):
                raw_plan_str = llm_response
            else:
                print(f"Warning: Unexpected LLM response type: {type(llm_response)}. Converting to string.")
                raw_plan_str = str(llm_response)

            # Parse the raw plan string
            current_plan = parse_plan(raw_plan_str)

            if not current_plan and raw_plan_str.strip():
                # Handle cases where parsing completely failed but LLM returned something
                last_error = "Failed to parse the generated plan structure."
                print(f"Attempt {attempt + 1} Error: {last_error}")
                continue # Retry if possible

            # Validate the parsed plan
            is_valid, issues = validator.validate(current_plan)

            if is_valid:
                print(f"Plan validated successfully on attempt {attempt + 1}.")
                return current_plan # Return the valid plan
            else:
                # Plan is not valid, format the issues and prepare for potential retry
                print(f"Attempt {attempt + 1} Error: Plan validation failed.")
                issue_descriptions = []
                for idx, desc in issues:
                    issue_descriptions.append(f"  - Step Index {idx if idx >= 0 else 'N/A'}: {desc}")
                last_error = f"Validation Issues:\n" + "\n".join(issue_descriptions)
                print(last_error)
                # If this was the last attempt, the loop will exit
                # Otherwise, it continues to the next iteration

        except Exception as e:
            print(f"Error during plan generation/parsing attempt {attempt + 1}: {e}")
            import traceback
            traceback.print_exc()
            last_error = f"Exception during generation/parsing: {e}"
            # Continue to the next retry iteration if attempts remain

    # If loop finishes without a valid plan
    print(f"Failed to generate a valid plan after {max_retries + 1} attempts.")
    # Return the last generated plan, even if invalid, or an empty list
    return current_plan

# Example Usage - Update to reflect new prompt/parsing
if __name__ == "__main__":
    import asyncio
    from langchain_community.llms.fake import FakeListLLM # For testing without real LLM
    # Assuming BaseTool and a way to create mock tools for testing
    from langchain_core.tools import tool as ToolDecorator

    # Mock Tools for testing
    @ToolDecorator
    def search_web_search(query: str) -> str:
        """Searches the web for information."""
        return f"Search results for: {query}" # Mock implementation

    @ToolDecorator
    def another_server_calculate(operand1: float, operand2: float, operation: str = 'add') -> float:
        """Performs calculation."""
        if operation == 'add': return operand1 + operand2
        return 0.0 # Mock implementation

    mock_tools = [search_web_search, another_server_calculate]

    async def run_planner_example():
        # --- Setup Fake LLM and Tools for testing ---
        # Example output matching the NEW prompt format
        fake_plan_output = """
Thought: First I need to find out what the ReWOO pattern is.
Tool Call: #E1 = search_web_search[{"query": "ReWOO agent pattern"}]
Expected Outcome: A description or definition of the ReWOO pattern.

Thought: Now that I have search results (referenced as #E1), I need to understand the core components mentioned. Let's assume the results mention Planner, Worker, Solver.
Expected Outcome: Identification of core ReWOO components.

Thought: I should calculate the complexity based on these components. Let's assume Planner=5, Worker=3, Solver=4. Need to add Worker and Solver first.
Tool Call: #E2 = another_server_calculate[{"operand1": 3, "operand2": 4, "operation": "add"}]
Expected Outcome: The sum of Worker and Solver complexity.

Thought: Adding the complexity of the worker and solver gives an intermediate result (#E2). Now add the planner complexity (5).
Tool Call: #E3 = another_server_calculate[{"operand1": "#E2", "operand2": 5, "operation": "add"}]
Expected Outcome: The total complexity calculated.

Thought: The total complexity is calculated (#E3). Now I can formulate the final answer based on the initial search (#E1) and calculations (#E3).
Expected Outcome: Final answer synthesizing the findings.
"""
        fake_llm = FakeListLLM(responses=[fake_plan_output])
        query = "Explain the ReWOO pattern and calculate its complexity based on components."
        # --- ---

        try:
            print(f"Running planner for query: '{query}'")
            plan_steps = await generate_plan(fake_llm, mock_tools, query)

            # Output should show the parsed steps from fake_plan_output matching new format
            assert len(plan_steps) == 5 # Updated expected number of steps
            assert plan_steps[0].step_index == 0
            assert plan_steps[0].tool_call is not None
            assert plan_steps[0].tool_call.tool_name == 'search_web_search'
            assert plan_steps[0].expected_outcome is not None
            assert plan_steps[1].tool_call is None # Step 2 has no tool call in example
            assert plan_steps[3].step_index == 3
            assert plan_steps[3].tool_call is not None
            assert plan_steps[3].tool_call.arguments['operand1'] == '#E2' # Check evidence reference

            print("\nPlanner example finished successfully with new structure.")

        except Exception as e:
            print(f"\nAn error occurred during planner example: {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(run_planner_example())

    # Remove old example code if it existed