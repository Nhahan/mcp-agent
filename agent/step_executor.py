# agent/step_executor.py
from typing import Dict, Any, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_mcp_adapters.base import MultiServerMCPClient

from .state import PlanStep
from .tool_selector import ToolSelector, ToolSelectionError
from .tool_input_generator import ToolInputGenerator, ToolInputGenerationError

class StepExecutionError(Exception):
    """Custom exception for errors during step execution."""
    pass

class StepExecutor:
    """Executes a single step of a plan, including tool selection, input generation, and execution."""

    def __init__(
        self,
        # Note: LLM might not be directly needed if selector/generator have it
        # llm: BaseLanguageModel, 
        mcp_client: MultiServerMCPClient,
        tool_selector: ToolSelector,
        tool_input_generator: ToolInputGenerator,
    ):
        """
        Initializes the StepExecutor.

        Args:
            mcp_client: Client for interacting with MCP server (fetching specs, executing tools).
            tool_selector: Instance for selecting tools.
            tool_input_generator: Instance for generating tool inputs.
        """
        # self.llm = llm
        self.mcp_client = mcp_client
        self.tool_selector = tool_selector
        self.tool_input_generator = tool_input_generator

    async def execute_step(self, plan_step: PlanStep) -> Dict[str, Any]:
        """
        Executes a single plan step.

        Args:
            plan_step: The plan step to execute.

        Returns:
            A dictionary containing the result of the step execution.
            Keys might include 'status', 'output', 'error'.

        Raises:
            StepExecutionError: If the step execution fails critically.
        """
        print(f"\n--- Executing Step {plan_step.step_index}: {plan_step.thought} ---")

        # If the step from the plan already specifies a tool, use it directly.
        # Otherwise, use the ToolSelector.
        selected_tool_name: Optional[str] = None
        selection_reasoning: Optional[str] = None

        if plan_step.tool_call and plan_step.tool_call.tool_name:
            # Scenario 1: Planner already decided the tool
            # TODO: Add validation maybe? Ensure pre-defined tool exists?
            # For now, assume the planner chose correctly if specified.
            print(f"Using pre-defined tool from plan: {plan_step.tool_call.tool_name}")
            selected_tool_name = plan_step.tool_call.tool_name
            # If arguments were also pre-defined by the planner, we might use them directly
            # But the ReWOO pattern usually involves generating inputs based on context.
            # Let's assume we always generate inputs for now, even if tool is pre-defined.

        elif plan_step.thought: # Only select tool if there's a thought process
            # Scenario 2: Need to select a tool based on thought/outcome
            try:
                print("Selecting tool...")
                selection_result = await self.tool_selector.select_tool(plan_step)
                selected_tool_name = selection_result['tool_name']
                selection_reasoning = selection_result['reasoning'] # Store for logging/context
                print(f"Tool Selector selected: {selected_tool_name}")
            except ToolSelectionError as e:
                error_msg = f"Failed to select tool for step {plan_step.step_index}: {e}"
                print(error_msg)
                return {"status": "failed", "error": error_msg}
            except Exception as e:
                error_msg = f"Unexpected error during tool selection for step {plan_step.step_index}: {e}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                return {"status": "failed", "error": error_msg}
        else:
            # Scenario 3: No tool call pre-defined, and no thought to select one.
            # This step likely doesn't involve a tool.
            print("Step does not require tool selection (no pre-defined tool or thought).")
            # TODO: Define how to handle non-tool steps. Maybe just return the thought?
            return {"status": "success", "output": plan_step.thought, "type": "thought_only"}

        # --- Input Generation --- #
        if selected_tool_name:
            try:
                print(f"Generating inputs for tool: {selected_tool_name}...")
                generated_inputs = await self.tool_input_generator.generate_inputs(
                    selected_tool_name=selected_tool_name,
                    plan_step=plan_step
                )
                print(f"Generated inputs: {generated_inputs}")
            except ToolInputGenerationError as e:
                error_msg = f"Failed to generate inputs for tool {selected_tool_name} (Step {plan_step.step_index}): {e}"
                print(error_msg)
                return {"status": "failed", "error": error_msg, "tool_name": selected_tool_name}
            except Exception as e:
                error_msg = f"Unexpected error during input generation for {selected_tool_name} (Step {plan_step.step_index}): {e}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                return {"status": "failed", "error": error_msg, "tool_name": selected_tool_name}
        else:
            # Should not happen if tool selection succeeded, but handle defensively.
             print("Error: Tool name was determined, but is now missing before input generation.")
             return {"status": "failed", "error": "Internal error: Tool name lost after selection."} 


        # --- Placeholder for Tool Execution (Task 8 onwards) --- #
        print(f"Executing tool '{selected_tool_name}' with inputs: {generated_inputs}...")
        # try:
        #     # tool_result = await self.mcp_client.execute_tool(
        #     #     tool_name=selected_tool_name,
        #     #     arguments=generated_inputs
        #     # )
        #     # print(f"Tool execution result: {tool_result}")
        #     # TODO: Process result into evidence (Task 8)
        #     tool_result = f"(Mock result for {selected_tool_name} with args {generated_inputs})"
        #     return {"status": "success", "output": tool_result, "tool_name": selected_tool_name, "inputs": generated_inputs}
        # except Exception as e:
        #     error_msg = f"Failed to execute tool {selected_tool_name} (Step {plan_step.step_index}): {e}"
        #     print(error_msg)
        #     return {"status": "failed", "error": error_msg, "tool_name": selected_tool_name, "inputs": generated_inputs}
        # --- End Placeholder ---

        # Mock return until execution is implemented
        mock_output = f"(Mock output: Would execute {selected_tool_name} with {generated_inputs})"
        print(f"Step {plan_step.step_index} finished (mock execution). Output: {mock_output}")
        return {
            "status": "success", # Assume success for now
            "output": mock_output,
            "tool_name": selected_tool_name,
            "inputs": generated_inputs,
            "selection_reasoning": selection_reasoning # Include if available
        }

# Example Usage (requires async context and mocks)
if __name__ == "__main__":
    import asyncio
    from langchain_community.llms.fake import FakeListLLM
    from .validation import InputValidator # Need validator for generator

    # --- Mock Components --- # 
    class MockMCPClient:
        async def get_all_tool_specifications(self):
            return {
                "search/web_search": {"description": "Searches web.", "parameters": {"query": {"type": "string", "required": True}}},
                "weather/get_current_weather": {"description": "Gets weather.", "parameters": {"location": {"type": "string", "required": True}}},
            }
        async def get_tool_specification(self, tool_name: str):
            all_specs = await self.get_all_tool_specifications()
            return all_specs.get(tool_name)
        # Mock execute_tool if needed for later tests
        # async def execute_tool(self, tool_name, arguments):
        #     return f"Executed {tool_name} with {arguments}"

    # Use fake LLM responses defined in other files for consistency if needed
    fake_llm_selector_response = '''
```json
{
  "reasoning": "Need to find info online.",
  "tool_name": "search/web_search"
}
```
    '''
    fake_llm_input_response = '''
```json
{
  "query": "What is the ReWOO pattern?"
}
```
    '''
    fake_llm = FakeListLLM(responses=[
        fake_llm_selector_response, # Response for tool selection
        fake_llm_input_response   # Response for input generation
    ])

    mock_client = MockMCPClient()
    mock_validator = InputValidator() # Use default validator
    mock_selector = ToolSelector(llm=fake_llm, mcp_client=mock_client)
    mock_input_generator = ToolInputGenerator(llm=fake_llm, mcp_client=mock_client, validator=mock_validator)

    step_executor = StepExecutor(
        mcp_client=mock_client,
        tool_selector=mock_selector,
        tool_input_generator=mock_input_generator
    )

    # --- Test Case 1: Step requires tool selection --- #
    plan_step_1 = PlanStep(
        step_index=0,
        thought="I need to research the ReWOO pattern online.",
        expected_outcome="A description of the ReWOO pattern.",
        status="pending",
        tool_call=None # Let selector choose
    )

    # --- Test Case 2: Step with pre-defined tool --- #
    plan_step_2 = PlanStep(
        step_index=1,
        thought="Get weather for London, as planned.",
        expected_outcome="London weather info.",
        status="pending",
        tool_call=ToolCall(tool_name="weather/get_current_weather", arguments={}) # Tool pre-defined
    )

    # --- Test Case 3: Step without tool call --- #
    plan_step_3 = PlanStep(
        step_index=2,
        thought="Synthesize the collected information.",
        expected_outcome="Final answer.",
        status="pending",
        tool_call=None
    )

    async def run_executor_example():
        print("--- Testing Step Executor --- ")

        result1 = await step_executor.execute_step(plan_step_1)
        print(f"\nResult Step 1: {result1}")
        assert result1['status'] == 'success'
        assert result1['tool_name'] == 'search/web_search'
        assert result1['inputs'] == {'query': 'What is the ReWOO pattern?'}

        # Need to reset LLM responses if the same instance is used
        # Or provide enough responses for all calls
        # For simplicity, let's assume separate LLM calls get next response in FakeListLLM
        # We need responses for step 2's input generation
        global fake_llm # Allow modifying the global fake_llm for demo
        fake_llm.responses = ['''
```json
{
  "location": "London, UK"
}
```
        ''']

        result2 = await step_executor.execute_step(plan_step_2)
        print(f"\nResult Step 2: {result2}")
        assert result2['status'] == 'success'
        assert result2['tool_name'] == 'weather/get_current_weather'
        assert result2['inputs'] == {'location': 'London, UK'}
        assert result2.get('selection_reasoning') is None # No selection was performed

        result3 = await step_executor.execute_step(plan_step_3)
        print(f"\nResult Step 3: {result3}")
        assert result3['status'] == 'success'
        assert result3['type'] == 'thought_only'
        assert result3['output'] == plan_step_3.thought


    asyncio.run(run_executor_example()) 