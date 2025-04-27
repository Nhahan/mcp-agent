from langchain_core.prompts import PromptTemplate
import json

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

# --- Tool Input Generation Prompts ---

# System prompt for the Tool Input Generator Agent
TOOL_INPUT_SYSTEM_PROMPT = """
You are an expert Tool Input Generator agent. Your task is to generate a valid JSON object containing the arguments required to call the specified tool, based on the current plan step and the tool's specification.

Analyze the 'Thought' and 'Expected Outcome' of the plan step to understand the context and required information.
Use the provided tool specification to identify the necessary parameters, their types, and any constraints.
Extract relevant information from the plan step context (thought, outcome, previous step evidence if available using #E<n>) to populate the arguments.

Selected Tool: {tool_name}
Tool Description: {tool_description}
Tool Specification (Parameters):
{tool_parameters_json}

Current Plan Step:
Thought: {plan_step_thought}
Expected Outcome: {plan_step_expected_outcome}
(Optional) Evidence from previous steps might be implicitly available in the thought process.

Instructions:
1. Carefully examine the tool's parameters (name, type, description, required).
2. Extract the necessary values for each required parameter from the plan step context.
3. Ensure the generated values match the expected parameter types (e.g., string, number, boolean).
4. If a parameter value comes from a previous step's output (e.g., mentioned as #E1 in thought), represent it as a string placeholder like "#E1". The execution engine will resolve this later.
5. Respond ONLY with a valid JSON object containing the arguments. The keys should be the parameter names, and the values should be the generated inputs.

Example Request:
Tool: weather/get_current_weather
Tool Description: Gets the current weather conditions for a specific location.
Tool Parameters: {{"location": {{"type": "string", "description": "The city and state/country (e.g., San Francisco, CA).", "required": true}}}}
Plan Step Thought: I need to get the weather for Paris, France.
Plan Step Expected Outcome: The weather in Paris.

Example Response Format:
```json
{{
  "location": "Paris, France"
}}
```

Example Request 2:
Tool: search/web_search
Tool Description: Searches the web.
Tool Parameters: {{"query": {{"type": "string", "description": "Search query", "required": true}}}}
Plan Step Thought: I need to search for the capital mentioned in #E1.
Plan Step Expected Outcome: Search results for the capital city.

Example Response Format 2:
```json
{{
  "query": "#E1"
}}
```
"""

# Prompt template for Tool Input Generation
TOOL_INPUT_PROMPT_TEMPLATE = PromptTemplate.from_template(
    TOOL_INPUT_SYSTEM_PROMPT
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
    print("\n--- Example Expected LLM Output (Selector) ---")
    print(expected_json_output.strip())

    # --- Tool Input Generator Example ---
    print("\n" + "-"*30 + "\n")

    selected_tool_name_inp = "weather/get_current_weather"
    selected_tool_desc_inp = "Gets the current weather conditions for a specific location."
    selected_tool_params_inp = {"location": {"type": "string", "description": "The city and state/country (e.g., San Francisco, CA).", "required": True}}
    example_thought_inp = "Okay, the selector chose the weather tool. Now I need the input for London."
    example_outcome_inp = "Arguments needed to call the weather tool for London."

    formatted_input_prompt = TOOL_INPUT_PROMPT_TEMPLATE.format(
        tool_name=selected_tool_name_inp,
        tool_description=selected_tool_desc_inp,
        tool_parameters_json=json.dumps(selected_tool_params_inp), # Parameters as JSON string
        plan_step_thought=example_thought_inp,
        plan_step_expected_outcome=example_outcome_inp
    )

    print("--- Formatted Tool Input Generator Prompt Example ---")
    print(formatted_input_prompt)

    # Example expected JSON output from the LLM for this input prompt:
    expected_input_json_output = """
```json
{
  "location": "London, UK"
}
```
    """
    print("\n--- Example Expected LLM Output (Input Generator) ---")
    print(expected_input_json_output.strip()) 