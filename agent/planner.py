# agent/planner.py
import re
import json
from typing import List, Optional

from langchain_core.language_models import BaseLanguageModel
# from langchain_core.prompts import PromptTemplate # Use the one from prompts module
from langchain_core.tools import BaseTool

from .state import PlanStep, ToolCall
# Import the specific prompt template
from .prompts.plan_prompts import PLANNER_PROMPT_TEMPLATE, PLANNER_REFINE_PROMPT_TEMPLATE
# Import the validator
from .validation import PlanValidator, ValidationIssue

# Remove old template definition
# PLANNER_TEMPLATE = """..."""

# Regular expression to parse the plan steps and tool calls based on the new prompt format
# It needs to capture Thought, optional Tool Call (#E<n> = Tool[Input]), and optional Expected Outcome
# Example Step:
# 1. Thought: I need to find the capital.
#    Tool Call: #E1 = search/web_search[{"query": "capital of France"}]
#    Expected Outcome: The capital city of France.
# Simpler Regex: Extract thought first, then look for optional Tool Call and Expected Outcome lines
# PLAN_STEP_REGEX = re.compile(r"Thought:\s*(.*?)(?:\n\s*Tool Call:\s*#E(\d+)\s*=\s*([\w_/]+)\[(.*)\])?(?:\n\s*Expected Outcome:\s*(.*?))?", re.DOTALL | re.IGNORECASE | re.MULTILINE)
# More robust parsing might be needed depending on LLM output variations.
# Let's try parsing line by line or using a dedicated parsing LLM if regex becomes too fragile.
# For now, stick to a simpler approach assuming the LLM follows instructions closely.
THOUGHT_REGEX = re.compile(r"Thought:\s*(.*)", re.IGNORECASE)
TOOL_CALL_REGEX = re.compile(r"Tool Call:\s*#E(\d+)\s*=\s*([\w_/]+)\[(.*)\]", re.IGNORECASE)
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

def parse_plan(plan_string: str) -> List[PlanStep]:
    """ Parses the LLM's plan string into a list of PlanStep objects based on the new format. """
    steps: List[PlanStep] = []
    current_step_index = 0 # Start step index at 0

    print(f"\n--- Raw Plan String ---\n{plan_string}\n----------------------") # Debugging raw output

    # Split into potential steps (e.g., based on step numbers or keywords like "Thought:")
    # A simple split by lines containing "Thought:" might work if format is consistent
    potential_step_blocks = re.split(r'\n(?=Thought:)', plan_string.strip(), flags=re.IGNORECASE)

    for block in potential_step_blocks:
        if not block.strip():
            continue

        thought_match = THOUGHT_REGEX.search(block)
        tool_call_match = TOOL_CALL_REGEX.search(block)
        expected_outcome_match = EXPECTED_OUTCOME_REGEX.search(block)

        thought = thought_match.group(1).strip() if thought_match else "" # Default to empty if not found
        tool_call = None
        expected_outcome = expected_outcome_match.group(1).strip() if expected_outcome_match else None

        if tool_call_match:
            step_num_str, tool_name, tool_input_str = tool_call_match.groups()
            tool_name = tool_name.strip()
            tool_input_str = tool_input_str.strip()
            try:
                # Attempt to parse input as JSON
                arguments = json.loads(tool_input_str)
                if not isinstance(arguments, dict):
                    print(f"Warning: Parsed non-dict JSON input for {tool_name}: {arguments}. Using raw string.")
                    arguments = {"input": tool_input_str} # Fallback
            except json.JSONDecodeError:
                # Treat as single string argument
                arguments = {"input": tool_input_str}
            except Exception as e:
                print(f"Error parsing tool arguments for {tool_name}: {e}. Input: {tool_input_str}")
                arguments = {"error": f"Argument parsing failed: {e}", "raw_input": tool_input_str}

            tool_call = ToolCall(tool_name=tool_name, arguments=arguments)

        if thought: # Only add a step if there is a thought process described
            steps.append(PlanStep(
                step_index=current_step_index,
                thought=thought,
                tool_call=tool_call,
                expected_outcome=expected_outcome,
                status="pending" # Initial status
            ))
            current_step_index += 1
        elif tool_call:
             # If there's a tool call but no thought, maybe log a warning or use a default thought?
             print(f"Warning: Tool call found without preceding thought in block: {block}")
             # Optionally add the step anyway, or skip?
             steps.append(PlanStep(
                step_index=current_step_index,
                thought="(Thought missing)", # Placeholder
                tool_call=tool_call,
                expected_outcome=expected_outcome,
                status="pending" # Initial status
            ))
             current_step_index += 1


    print(f"\n--- Parsed Plan Steps ({len(steps)} steps) ---") # Debugging parsed output
    for step in steps:
        print(f"Step {step.step_index}:")
        print(f"  Thought: {step.thought}")
        if step.tool_call:
            print(f"  Tool Call: {step.tool_call.tool_name}[{step.tool_call.arguments}]")
        if step.expected_outcome:
            print(f"  Expected Outcome: {step.expected_outcome}")
    print("--------------------------")

    if not steps and plan_string.strip():
         print("Warning: Could not parse any steps from the plan string. Returning empty plan.")
         # Consider adding raw string as a single thought step or raising error

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