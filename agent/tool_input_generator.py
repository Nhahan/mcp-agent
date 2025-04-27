# agent/tool_input_generator.py
import json
import re
from typing import Dict, Any, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_mcp_adapters.base import MultiServerMCPClient # Assuming this provides tool specs

from .state import PlanStep # Assuming PlanStep is defined
from .prompts.tool_prompts import TOOL_INPUT_PROMPT_TEMPLATE
# Import the validator
from .validation import InputValidator, InputValidationIssue

class ToolInputGenerationError(Exception):
    """Custom exception for tool input generation errors."""
    pass

class ToolInputGenerator:
    """Generates tool inputs based on plan step, tool specs, using an LLM."""

    def __init__(
        self,
        llm: BaseLanguageModel,
        mcp_client: MultiServerMCPClient,
        max_retries: int = 1,
        validator: Optional[InputValidator] = None # Make validator optional or provide default
    ):
        """
        Initializes the ToolInputGenerator.

        Args:
            llm: The language model instance for generating inputs.
            mcp_client: MCP client to get tool specifications.
            max_retries: Max retries if input generation or validation fails.
            validator: An instance for validating generated inputs (Task 7.5).
        """
        self.llm = llm
        self.mcp_client = mcp_client
        self.max_retries = max_retries
        self.prompt_template = TOOL_INPUT_PROMPT_TEMPLATE
        self.validator = validator or InputValidator() # Initialize validator

    async def _get_tool_specification(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Fetches the specification for a specific tool."""
        try:
            # Assume client has a method like get_tool_specification(tool_name)
            spec = await self.mcp_client.get_tool_specification(tool_name) # Placeholder method
            return spec
        except Exception as e:
            print(f"Error fetching specification for tool '{tool_name}': {e}")
            return None

    def _parse_input_response(self, response_content: str) -> Dict[str, Any]:
        """Parses the LLM response (expected JSON) to extract tool arguments."""
        try:
            # Extract JSON from potential markdown code blocks
            match = re.search(r'```json\n({.*?})\n```', response_content, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                json_str = response_content.strip()

            parsed_json = json.loads(json_str)

            if not isinstance(parsed_json, dict):
                raise ValueError("Response is not a JSON object.")

            # Basic validation passed, detailed validation happens later (Task 7.5)
            return parsed_json

        except json.JSONDecodeError as e:
            raise ToolInputGenerationError(f"Failed to parse LLM response as JSON: {e}\nResponse: {response_content}") from e
        except ValueError as e:
             raise ToolInputGenerationError(f"Invalid response format: {e}\nResponse: {response_content}") from e
        except Exception as e:
            raise ToolInputGenerationError(f"Unexpected error parsing input response: {e}\nResponse: {response_content}") from e

    async def generate_inputs(self, selected_tool_name: str, plan_step: PlanStep) -> Dict[str, Any]:
        """
        Generates validated inputs for the selected tool based on the plan step.

        Args:
            selected_tool_name: The name of the tool selected in the previous stage.
            plan_step: The current plan step providing context.

        Returns:
            A dictionary representing the validated arguments for the tool.

        Raises:
            ToolInputGenerationError: If valid inputs cannot be generated after retries.
        """
        last_error: Optional[Exception] = None

        tool_spec = await self._get_tool_specification(selected_tool_name)
        if not tool_spec:
            raise ToolInputGenerationError(f"Could not retrieve specification for tool '{selected_tool_name}'. Cannot generate inputs.")

        tool_description = tool_spec.get('description', 'No description available.')
        # Parameters might be nested differently, adjust based on actual spec structure
        tool_parameters = tool_spec.get('parameters', {})
        tool_parameters_json = json.dumps(tool_parameters) # Pass parameters as JSON string to prompt

        for attempt in range(self.max_retries + 1):
            print(f"\n--- Tool Input Generation Attempt {attempt + 1}/{self.max_retries + 1} for {selected_tool_name} ---")
            try:
                # Prepare prompt
                prompt_input = {
                    "tool_name": selected_tool_name,
                    "tool_description": tool_description,
                    "tool_parameters_json": tool_parameters_json,
                    "plan_step_thought": plan_step.thought or "(No thought provided)",
                    "plan_step_expected_outcome": plan_step.expected_outcome or "(No expected outcome provided)",
                }
                generation_chain = self.prompt_template | self.llm

                # Invoke LLM
                llm_response = await generation_chain.ainvoke(prompt_input)

                # Extract content
                if hasattr(llm_response, 'content'):
                    response_content = llm_response.content
                elif isinstance(llm_response, str):
                    response_content = llm_response
                else:
                    print(f"Warning: Unexpected LLM response type for input generation: {type(llm_response)}. Converting to string.")
                    response_content = str(llm_response)

                print(f"Raw LLM Response:\n{response_content}") # Debug output

                # Parse response
                generated_inputs = self._parse_input_response(response_content)
                print(f"Parsed Inputs: {generated_inputs}")

                # --- Integrate Input Validation --- 
                is_valid, validation_issues = self.validator.validate(
                    tool_parameters, generated_inputs # Pass spec and generated inputs
                )
                if not is_valid:
                    # Format issues for error message and potential refinement prompt
                    issue_desc = "; ".join([f"{name}: {desc}" for name, desc in validation_issues])
                    error_msg = f"Generated inputs failed validation: {issue_desc}"
                    raise ToolInputGenerationError(error_msg)
                # --- Validation Successful --- 

                print(f"Inputs generated and validated successfully for {selected_tool_name}.")
                return generated_inputs # Return validated inputs

            except ToolInputGenerationError as e:
                print(f"Attempt {attempt + 1} Error: {e}")
                last_error = e
                # TODO: Optionally implement refinement logic here
                # Could involve passing validation_issues back to LLM with a new prompt
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Attempt {attempt + 1} Unexpected Error: {e}")
                last_error = ToolInputGenerationError(f"Unexpected error during input generation: {e}") from e

            # Retry logic
            if attempt < self.max_retries:
                print("Retrying input generation...")
            else:
                print("Max retries reached.")

        # If loop finishes
        raise ToolInputGenerationError(f"Failed to generate valid inputs for tool '{selected_tool_name}' after maximum retries.") from last_error

# Example Usage (requires async context and mocks)
if __name__ == "__main__":
    import asyncio
    from langchain_community.llms.fake import FakeListLLM

    # Mock MCP Client with specific tool spec method
    class MockMCPClient:
        async def get_tool_specification(self, tool_name: str):
            specs = {
                "weather/get_current_weather": {
                    "description": "Gets the current weather for a location.",
                    "parameters": {"location": {"type": "string", "description": "The city (e.g., London)", "required": True}}
                },
                 "search/web_search": {
                    "description": "Searches the web.",
                    "parameters": {"query": {"type": "string", "required": True}}
                }
            }
            return specs.get(tool_name)

    # Mock Plan Step
    mock_plan_step = PlanStep(
        step_index=1, # Assume this step follows a previous one
        thought="Need to get the weather for Berlin, Germany, which was mentioned in #E1 as the location.",
        expected_outcome="The weather conditions in Berlin.",
        status="pending"
    )
    selected_tool = "weather/get_current_weather"

    async def run_generator_example():
        # Fake LLM response matching the expected JSON format
        fake_llm_response_inputs = '''
```json
{
  "location": "Berlin, Germany"
}
```
        '''
        # Second response for retry (e.g., missing required field initially)
        fake_llm_invalid_response = '''
```json
{
  "city": "Berlin" 
}
```
        '''

        # Test successful generation
        print("--- Testing Successful Input Generation & Validation ---")
        fake_llm_success = FakeListLLM(responses=[fake_llm_response_inputs])
        mock_client = MockMCPClient()
        # Now includes the validator implicitly
        generator_success = ToolInputGenerator(llm=fake_llm_success, mcp_client=mock_client)
        try:
            inputs = await generator_success.generate_inputs(selected_tool, mock_plan_step)
            print(f"Success! Generated & Validated Inputs: {inputs}")
            assert inputs == {"location": "Berlin, Germany"}
        except ToolInputGenerationError as e:
            print(f"Error during successful generation test: {e}")

        # Test generation failure due to validation
        print("\n--- Testing Input Generation Failure (Validation) with Retry ---")
        # LLM first returns invalid inputs (missing required 'location'), then valid
        invalid_input_response = '''
```json
{
  "unit": "celsius" 
}
```
        '''
        fake_llm_validation_retry = FakeListLLM(responses=[invalid_input_response, fake_llm_response_inputs])
        generator_validation_retry = ToolInputGenerator(llm=fake_llm_validation_retry, mcp_client=mock_client, max_retries=1)
        try:
            inputs = await generator_validation_retry.generate_inputs(selected_tool, mock_plan_step)
            print(f"Retry Success! Generated & Validated Inputs: {inputs}")
            assert inputs == {"location": "Berlin, Germany"}
        except ToolInputGenerationError as e:
            print(f"Error during validation retry test: {e}") # Expected to succeed on retry

        # Test parsing failure with retry
        print("\n--- Testing Input Generation Failure (Parsing) with Retry ---")
        malformed_json_response = "```json\n{\n  "location\": \"Malformed JSON\",\n```"
        fake_llm_parse_retry = FakeListLLM(responses=[malformed_json_response, fake_llm_response_inputs])
        generator_parse_retry = ToolInputGenerator(llm=fake_llm_parse_retry, mcp_client=mock_client, max_retries=1)
        try:
            inputs = await generator_parse_retry.generate_inputs(selected_tool, mock_plan_step)
            print(f"Retry Success! Generated Inputs (after parsing error): {inputs}")
            assert inputs == {"location": "Berlin, Germany"}
        except ToolInputGenerationError as e:
            print(f"Error during parsing retry test: {e}")

        # Test failure (max retries exceeded due to validation)
        print("\n--- Testing Input Generation Failure (Max Retries - Validation) ---")
        fake_llm_validation_fail = FakeListLLM(responses=[invalid_input_response] * 2) # Always invalid
        generator_validation_fail = ToolInputGenerator(llm=fake_llm_validation_fail, mcp_client=mock_client, max_retries=1)
        try:
            await generator_validation_fail.generate_inputs(selected_tool, mock_plan_step)
            print("Error: Input generation succeeded unexpectedly!")
        except ToolInputGenerationError as e:
            print(f"Successfully caught expected error: {e}")
            assert "Failed to generate valid inputs" in str(e)

    asyncio.run(run_generator_example()) 