# agent/tool_selector.py
import json
import re # Import the re module
from typing import List, Dict, Any, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_mcp_adapters.base import MultiServerMCPClient # Assuming this client provides tool specs

from .state import PlanStep # Assuming PlanStep is defined in agent.state
from .prompts.tool_prompts import TOOL_SELECTOR_PROMPT_TEMPLATE

class ToolSelectionError(Exception):
    """Custom exception for tool selection errors."""
    pass

class ToolSelector:
    """Selects the appropriate tool based on the current plan step using an LLM."""

    def __init__(
        self,
        llm: BaseLanguageModel,
        mcp_client: MultiServerMCPClient, # Client to get available tools
        max_retries: int = 1
    ):
        """
        Initializes the ToolSelector.

        Args:
            llm: The language model instance for selecting tools.
            mcp_client: The MCP client instance to fetch available tool specifications.
            max_retries: Maximum attempts to retry tool selection if validation fails.
        """
        self.llm = llm
        self.mcp_client = mcp_client
        self.max_retries = max_retries
        self.prompt_template = TOOL_SELECTOR_PROMPT_TEMPLATE

    async def _get_available_tools_formatted(self) -> str:
        """Fetches available tools from the MCP client and formats them for the prompt."""
        try:
            # Assuming the client has a method like get_all_tool_specifications()
            # or similar to retrieve details needed for the prompt.
            # The exact method name might differ based on langchain-mcp-adapters implementation.
            # We need tool name, description, and potentially parameters.
            available_tools = await self.mcp_client.get_all_tool_specifications() # Placeholder method

            if not available_tools:
                return "No tools available."

            # Format tools for the prompt (adjust based on actual spec structure)
            formatted_list = []
            for tool_name, spec in available_tools.items():
                # Example formatting, adjust based on the actual spec object
                params = spec.get('parameters', {})
                param_str = json.dumps(params)
                formatted_list.append(f"- {tool_name}: {spec.get('description', 'No description')} (Args: {param_str})")

            return "\n".join(formatted_list)

        except Exception as e:
            print(f"Error fetching or formatting tool specifications: {e}")
            # Fallback or re-raise depending on desired behavior
            return "Error retrieving tool list."

    def _parse_selection_response(self, response_content: str) -> Dict[str, str]:
        """Parses the LLM response (expected JSON) to extract tool name and reasoning."""
        try:
            # LLM might wrap JSON in ```json ... ```, try to extract it
            match = re.search(r'```json\n({.*?})\n```', response_content, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                # Assume the response is just the JSON string
                json_str = response_content.strip()

            parsed_json = json.loads(json_str)

            if not isinstance(parsed_json, dict) or \
               'tool_name' not in parsed_json or \
               'reasoning' not in parsed_json:
                raise ValueError("Response JSON missing required keys ('tool_name', 'reasoning').")

            # Basic validation
            if not isinstance(parsed_json['tool_name'], str) or not parsed_json['tool_name']:
                 raise ValueError("'tool_name' must be a non-empty string.")
            if not isinstance(parsed_json['reasoning'], str):
                 raise ValueError("'reasoning' must be a string.")

            return {
                "tool_name": parsed_json['tool_name'].strip(),
                "reasoning": parsed_json['reasoning'].strip()
            }
        except json.JSONDecodeError as e:
            raise ToolSelectionError(f"Failed to parse LLM response as JSON: {e}\nResponse: {response_content}") from e
        except ValueError as e:
            raise ToolSelectionError(f"Invalid response format: {e}\nResponse: {response_content}") from e
        except Exception as e:
            # Catch any other unexpected errors during parsing
            raise ToolSelectionError(f"Unexpected error parsing selection response: {e}\nResponse: {response_content}") from e

    async def select_tool(self, plan_step: PlanStep) -> Dict[str, str]:
        """
        Selects the best tool for the given plan step.

        Args:
            plan_step: The current plan step containing thought and expected outcome.

        Returns:
            A dictionary containing the selected 'tool_name' and 'reasoning'.

        Raises:
            ToolSelectionError: If a valid tool cannot be selected after retries.
        """
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            print(f"\n--- Tool Selection Attempt {attempt + 1}/{self.max_retries + 1} ---")
            try:
                available_tools_str = await self._get_available_tools_formatted()
                if "Error retrieving tool list" in available_tools_str or "No tools available" in available_tools_str:
                     raise ToolSelectionError(f"Cannot select tool: {available_tools_str}")

                # Prepare prompt
                prompt_input = {
                    "available_tools": available_tools_str,
                    "plan_step_thought": plan_step.thought or "(No thought provided)",
                    "plan_step_expected_outcome": plan_step.expected_outcome or "(No expected outcome provided)",
                }
                selection_chain = self.prompt_template | self.llm

                # Invoke LLM
                llm_response = await selection_chain.ainvoke(prompt_input)

                # Extract content safely
                if hasattr(llm_response, 'content'):
                    response_content = llm_response.content
                elif isinstance(llm_response, str):
                    response_content = llm_response
                else:
                    print(f"Warning: Unexpected LLM response type for tool selection: {type(llm_response)}. Converting to string.")
                    response_content = str(llm_response)

                print(f"Raw LLM Response:\n{response_content}") # Debug output

                # Parse response
                selection = self._parse_selection_response(response_content)
                selected_tool_name = selection['tool_name']

                # Validate against available tools from MCP client
                # Re-fetch or use cached list if available
                # Note: _get_available_tools_formatted fetches specs, maybe client has a simpler list_tool_names()
                available_tool_specs = await self.mcp_client.get_all_tool_specifications() # Placeholder
                if selected_tool_name not in available_tool_specs:
                    raise ToolSelectionError(
                        f"Validation Failed: Selected tool '{selected_tool_name}' is not among available tools: {list(available_tool_specs.keys())}"
                    )

                print(f"Tool '{selected_tool_name}' selected successfully. Reasoning: {selection['reasoning']}")
                return selection # Return the validated selection

            except ToolSelectionError as e:
                print(f"Attempt {attempt + 1} Error: {e}")
                last_error = e
            except Exception as e:
                # Catch unexpected errors during LLM call or validation
                import traceback
                traceback.print_exc()
                print(f"Attempt {attempt + 1} Unexpected Error: {e}")
                last_error = ToolSelectionError(f"Unexpected error during tool selection: {e}") from e

            # If loop continues, it means an error occurred
            if attempt < self.max_retries:
                print("Retrying tool selection...")
            else:
                print("Max retries reached.")

        # If loop finishes without success
        raise ToolSelectionError("Failed to select a valid tool after maximum retries.") from last_error

# Example Usage (requires async context and mocks)
if __name__ == "__main__":
    import asyncio
    import re # Ensure re is imported for example usage
    from langchain_community.llms.fake import FakeListLLM

    # Mock MCP Client
    class MockMCPClient:
        async def get_all_tool_specifications(self):
            # Return example tool specs similar to what the real client might provide
            return {
                "search/web_search": {
                    "description": "Searches the web for general information.",
                    "parameters": {"query": {"type": "string"}}
                },
                "weather/get_current_weather": {
                    "description": "Gets the current weather for a location.",
                    "parameters": {"location": {"type": "string"}}
                }
            }

    # Mock Plan Step
    mock_plan_step = PlanStep(
        step_index=0,
        thought="Find the weather in Paris.",
        expected_outcome="Current weather conditions in Paris, France.",
        status="pending"
    )

    async def run_selector_example():
        # Fake LLM response matching the expected JSON format
        fake_llm_response = '''
```json
{
  "reasoning": "The user wants the current weather for Paris. The 'weather/get_current_weather' tool is specifically designed for this.",
  "tool_name": "weather/get_current_weather"
}
```
        '''
        # Second fake response for a retry scenario (e.g., selects invalid tool first)
        fake_llm_invalid_response = '''
```json
{
  "reasoning": "Searching seems appropriate.",
  "tool_name": "invalid_tool_name" 
}
```
        '''

        # Test successful selection
        print("--- Testing Successful Selection ---")
        fake_llm_success = FakeListLLM(responses=[fake_llm_response])
        mock_client = MockMCPClient()
        selector_success = ToolSelector(llm=fake_llm_success, mcp_client=mock_client)
        try:
            selection = await selector_success.select_tool(mock_plan_step)
            print(f"Success! Selected Tool: {selection['tool_name']}, Reasoning: {selection['reasoning']}")
            assert selection['tool_name'] == "weather/get_current_weather"
        except ToolSelectionError as e:
            print(f"Error during successful selection test: {e}")

        # Test selection failure with retry
        print("\n--- Testing Selection Failure with Retry ---")
        fake_llm_retry = FakeListLLM(responses=[fake_llm_invalid_response, fake_llm_response]) # Invalid then valid
        selector_retry = ToolSelector(llm=fake_llm_retry, mcp_client=mock_client, max_retries=1)
        try:
            selection = await selector_retry.select_tool(mock_plan_step)
            print(f"Retry Success! Selected Tool: {selection['tool_name']}, Reasoning: {selection['reasoning']}")
            assert selection['tool_name'] == "weather/get_current_weather"
        except ToolSelectionError as e:
            print(f"Error during retry test: {e}")

        # Test selection failure (max retries exceeded)
        print("\n--- Testing Selection Failure (Max Retries) ---")
        fake_llm_fail = FakeListLLM(responses=[fake_llm_invalid_response] * 2) # Always invalid
        selector_fail = ToolSelector(llm=fake_llm_fail, mcp_client=mock_client, max_retries=1)
        try:
            await selector_fail.select_tool(mock_plan_step)
            print("Error: Selection succeeded unexpectedly!")
        except ToolSelectionError as e:
            print(f"Successfully caught expected error: {e}")
            assert "Failed to select a valid tool after maximum retries" in str(e)

    asyncio.run(run_selector_example()) 