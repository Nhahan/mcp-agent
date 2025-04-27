from langchain_core.prompts import PromptTemplate

# System prompt for the Tool Selector Agent
TOOL_SELECTOR_SYSTEM_PROMPT = """
You are an expert Tool Selector agent. Your task is to choose the single most appropriate tool from the available list to achieve the objective described in the current plan step.

Analyze the 'Thought' and 'Expected Outcome' of the current plan step carefully. Compare the required action with the capabilities of the available tools described below.

Available Tools:
{available_tools}

Current Plan Step:
Thought: {plan_step_thought}
Expected Outcome: {plan_step_expected_outcome}

Instructions:
1. Identify the core task required by the plan step.
2. Evaluate which available tool best matches this task based on its description and purpose.
3. Provide a brief reasoning for your choice, explaining why the selected tool is suitable and others might be less so.
4. Respond ONLY with a JSON object containing the following keys:
   - "reasoning": Your explanation for selecting the tool.
   - "tool_name": The exact name of the selected tool (e.g., 'server/tool_name').

Example Response Format:
```json
{{
  "reasoning": "The plan step requires searching the web for recent information, and the 'search/web_search' tool is designed specifically for this purpose.",
  "tool_name": "search/web_search"
}}
```
"""

# Prompt template for Tool Selection
TOOL_SELECTOR_PROMPT_TEMPLATE = PromptTemplate.from_template(
    TOOL_SELECTOR_SYSTEM_PROMPT
)

# Example Usage (for testing and demonstration)
if __name__ == "__main__":
    example_thought = "I need to find the current weather in London."
    example_outcome = "The current temperature and weather conditions in London."
    example_tools_list = [
        {
            "name": "search/web_search",
            "description": "Searches the web for general information.",
            "parameters": {"query": {"type": "string", "description": "The search query."}}
        },
        {
            "name": "weather/get_current_weather",
            "description": "Gets the current weather conditions for a specific location.",
            "parameters": {"location": {"type": "string", "description": "The city and state/country (e.g., San Francisco, CA)."}}
        },
        {
            "name": "knowledge/lookup",
            "description": "Looks up static facts in a knowledge base.",
            "parameters": {"entity": {"type": "string", "description": "The entity to look up."}}
        }
    ]

    # Format available tools for the prompt
    formatted_tools = "\n".join([
        f"- {tool['name']}: {tool['description']} (Args: {tool['parameters']})"
        for tool in example_tools_list
    ])

    # Format the complete prompt
    formatted_prompt = TOOL_SELECTOR_PROMPT_TEMPLATE.format(
        available_tools=formatted_tools,
        plan_step_thought=example_thought,
        plan_step_expected_outcome=example_outcome
    )

    print("--- Formatted Tool Selector Prompt Example ---")
    print(formatted_prompt)

    # Example expected JSON output from the LLM for this prompt:
    expected_json_output = """
```json
{
  "reasoning": "The plan step specifically asks for current weather information for a location (London). The 'weather/get_current_weather' tool is designed for exactly this task, whereas web search is too general and knowledge lookup is for static facts.",
  "tool_name": "weather/get_current_weather"
}
```
    """
    print("\n--- Example Expected LLM Output ---")
    print(expected_json_output.strip()) 