# agent/planner.py
import re
import json
from typing import List, Dict, Any, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool

from .state import PlanStep, ToolCall

# Basic prompt template for the Planner
# Needs to instruct the model on the ReWOO format (Plan:, #E1 = Tool[Input], etc.)
# And provide the available tools with descriptions.
PLANNER_TEMPLATE = """
Given the user query and the available tools, create a step-by-step plan to answer the query.
For each step, explain your reasoning (Plan:) and identify the tool and input needed for that step using the format #E<step_number> = ToolName[Input].
The input for a tool should be a valid JSON string if the tool expects multiple arguments, or a single string if it expects one.
If a step doesn't require a tool, just provide the reasoning (Plan:).
Refer to the outputs of previous steps using #E<step_number> as variables in the input for subsequent steps if needed.
Base your final answer ONLY on the evidence gathered through tool use.

Available Tools:
{tool_descriptions}

User Query: {query}

Plan:
"""

# Regular expression to parse the plan steps and tool calls
# Matches "Plan: <thought>" and optionally "#E<n> = <tool_name>[<json_or_string_input>]"
PLAN_STEP_REGEX = re.compile(r"Plan:\s*(.*?)(?:\n#E(\d+)\s*=\s*([\w_/]+)\[(.*)\])?", re.DOTALL)

def _format_tool_descriptions(tools: List[BaseTool]) -> str:
    """ Formats the tool descriptions for the planner prompt. """
    descriptions = []
    for tool in tools:
        # Include tool name, description, and args schema
        schema_desc = tool.args_schema.schema() if tool.args_schema else 'No arguments'
        descriptions.append(f"- {tool.name}: {tool.description}\n  Args: {schema_desc}")
    return "\n".join(descriptions)

def parse_plan(plan_string: str) -> List[PlanStep]:
    """ Parses the LLM's plan string into a list of PlanStep objects. """
    steps: List[PlanStep] = []
    matches = PLAN_STEP_REGEX.findall(plan_string)

    print(f"\n--- Raw Plan String ---\n{plan_string}\n----------------------") # Debugging raw output

    step_index_counter = 1 # Start evidence counter at 1
    for match in matches:
        thought, step_num_str, tool_name, tool_input_str = match

        thought = thought.strip()
        tool_call: Optional[ToolCall] = None

        if tool_name and tool_input_str:
            tool_name = tool_name.strip()
            tool_input_str = tool_input_str.strip()

            # Basic validation of the step number if needed, though regex captures it
            # step_num = int(step_num_str) if step_num_str else None
            # if step_num != step_index_counter:
            #    print(f"Warning: Plan step number mismatch. Expected {step_index_counter}, got {step_num}")
            #    # Decide how to handle mismatch - use regex num, counter, or raise error

            try:
                # Attempt to parse input as JSON, otherwise treat as string
                # Handle potential escaped characters in LLM output if necessary
                tool_input_str_cleaned = tool_input_str # Add cleaning if needed
                arguments = json.loads(tool_input_str_cleaned)
                if not isinstance(arguments, dict):
                     # If JSON parses but isn't a dict, treat as single string arg if tool expects one?
                     # For now, assume if JSON parsable, it should be a dict for kwargs
                     print(f"Warning: Parsed non-dict JSON input for {tool_name}: {arguments}. Treating as raw string.")
                     arguments = {"input": tool_input_str} # Fallback or specific handling needed
            except json.JSONDecodeError:
                # If not valid JSON, treat as a single string argument (common case)
                # We might need tool schema here to know if the tool expects a single string or dict
                # For simplicity now, assume non-JSON is single 'input' arg or handled by tool
                arguments = {"input": tool_input_str} # Or pass raw string if tool expects it

            tool_call = ToolCall(tool_name=tool_name, arguments=arguments)
            step_index_counter += 1 # Increment only if a tool call was parsed

        if thought or tool_call: # Add step only if there's thought or a tool call
            steps.append(PlanStep(thought=thought, tool_call=tool_call))

    print(f"\n--- Parsed Plan Steps ({len(steps)} steps) ---") # Debugging parsed output
    for i, step in enumerate(steps):
        print(f"Step {i+1}:")
        print(f"  Thought: {step['thought']}")
        if step['tool_call']:
            print(f"  Tool Call: {step['tool_call']['tool_name']}[{step['tool_call']['arguments']}]")
    print("--------------------------")

    if not steps and plan_string:
         print("Warning: Could not parse any steps from the plan string. Returning empty plan.")
         # Optionally add the raw string as a single thought step?
         # steps.append(PlanStep(thought=plan_string.strip(), tool_call=None))


    return steps


async def generate_plan(
    llm: BaseLanguageModel,
    tools: List[BaseTool],
    query: str
) -> List[PlanStep]:
    """
    Generates a plan using the LLM based on the query and available tools,
    then parses the plan into structured steps.

    Args:
        llm: The language model instance to use for planning.
        tools: A list of available Langchain tools.
        query: The user's input query.

    Returns:
        A list of PlanStep objects representing the generated plan.
    """
    tool_descriptions = _format_tool_descriptions(tools)
    prompt = PromptTemplate.from_template(PLANNER_TEMPLATE)
    planner_chain = prompt | llm

    print("\nGenerating plan...")
    # Use async invocation if LLM supports it
    if hasattr(planner_chain, 'ainvoke'):
        raw_plan = await planner_chain.ainvoke({"tool_descriptions": tool_descriptions, "query": query})
    else:
         # Fallback to sync invoke if necessary
         raw_plan = planner_chain.invoke({"tool_descriptions": tool_descriptions, "query": query})


    if isinstance(raw_plan, dict) and hasattr(raw_plan, 'content'): # Handle AIMessage like objects
         raw_plan_str = raw_plan.content
    elif isinstance(raw_plan, str):
         raw_plan_str = raw_plan
    else:
         # Handle other potential response types from LLM Chain
         print(f"Warning: Unexpected raw plan type: {type(raw_plan)}. Attempting to convert to string.")
         raw_plan_str = str(raw_plan)


    # Parse the raw plan string
    parsed_plan = parse_plan(raw_plan_str)
    return parsed_plan

# Example Usage
if __name__ == "__main__":
    import asyncio
    from langchain_community.llms.fake import FakeListLLM # For testing without real LLM
    from agent.tools.registry import ToolRegistry # Assuming registry is set up

    async def run_planner_example():
        # --- Setup Fake LLM and Tools for testing ---
        fake_plan_output = """
Plan: First I need to find out what the ReWOO pattern is.
#E1 = web_search_placeholder/search[{"query": "ReWOO agent pattern"}]
Plan: Now that I have search results, I need to understand the core components mentioned in the results. Let's assume the results mention Planner, Worker, Solver.
Plan: I should calculate the complexity based on these components. Let's assume Planner=5, Worker=3, Solver=4.
#E2 = another_server/calculate[{"operand1": 5, "operand2": 3, "operation": "add"}]
Plan: Adding the complexity of the worker and solver gives an intermediate result. Now add the planner complexity.
#E3 = another_server/calculate[{"operand1": #E2, "operand2": 4, "operation": "add"}]
Plan: The total complexity is calculated. Now I can formulate the final answer based on the initial search and calculations.
"""
        fake_llm = FakeListLLM(responses=[fake_plan_output])
        registry = ToolRegistry() # Uses mock executor by default
        tools = registry.get_tools()
        query = "Explain the ReWOO pattern and calculate its complexity based on components."
        # --- ---

        try:
            print(f"Running planner for query: '{query}'")
            plan_steps = await generate_plan(fake_llm, tools, query)

            # Output should show the parsed steps from fake_plan_output
            assert len(plan_steps) > 0
            assert plan_steps[0]['tool_call']['tool_name'] == 'web_search_placeholder/search'
            assert plan_steps[2]['tool_call']['arguments']['operand1'] == '#E2' # Check variable substitution placeholder

            print("\nPlanner example finished successfully.")

        except Exception as e:
            print(f"\nAn error occurred during planner example: {e}")

    asyncio.run(run_planner_example()) 