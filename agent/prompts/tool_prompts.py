from langchain_core.prompts import PromptTemplate
import json

# Simplified prompt for Tool Selection
TOOL_SELECTOR_PROMPT_TEMPLATE = PromptTemplate.from_template(
    """Choose the single best tool for the plan step.
Respond ONLY with JSON: {{"reasoning": "<why_this_tool>", "tool_name": "<tool_name>"}}. No other text.

Available Tools:
{available_tools}

Plan Step:
Thought: {plan_step_thought}
Expected Outcome: {plan_step_expected_outcome}

Instructions:
1. Identify the core task in the Plan Step.
2. Select the best matching tool from Available Tools.
3. Provide brief reasoning.
4. Output ONLY the JSON: {{"reasoning": "...", "tool_name": "..."}}.

Output the JSON tool selection now.
"""
)

# Simplified prompt for Tool Input Generation
TOOL_INPUT_PROMPT_TEMPLATE = PromptTemplate.from_template(
    """Generate the JSON arguments for the selected tool based on the plan step and tool spec.
Respond ONLY with the valid JSON arguments object. No other text.

Selected Tool: {tool_name}
Tool Description: {tool_description}
Tool Parameters:
{tool_parameters_json}

Plan Step:
Thought: {plan_step_thought}
Expected Outcome: {plan_step_expected_outcome}
(Use #E<n> placeholder for prior step evidence if needed in arguments)

Instructions:
1. **IMPORTANT**: Any examples provided in prompts are for illustrating the **format** ONLY. Do NOT use the *content* of examples when generating arguments.
2. Carefully analyze the 'Plan Step' (Thought and Expected Outcome) provided above to identify the values needed for the tool's parameters.
3. Extract the required argument values **strictly and solely** from the 'Plan Step' context (Thought and Expected Outcome) provided above. **Do NOT use information from the original query, tool descriptions, or invent values.**
4. If a value should come from a previous step's result (indicated by #E<n> in the Plan Step thought or expected outcome), use the exact string "#E<n>" as the argument value.
5. Ensure the extracted values match the expected data types specified in 'Tool Parameters'.
6. Construct a valid JSON object containing only the required arguments and their extracted values.
7. If the tool requires no arguments according to its parameters, output an empty JSON object: {{}}.
8. Output ONLY the valid JSON arguments object. No other text, explanations, or markdown formatting.

Output the JSON arguments now.
"""
)

# Example Usage (for testing)
if __name__ == "__main__":
    example_thought = "Get weather in London."
    example_outcome = "Current weather."
    example_tools_list = [
        {"name": "search/web_search", "description": "Web search.", "parameters": {"query": {"type": "string"}}},
        {"name": "weather/get_current_weather", "description": "Get weather.", "parameters": {"location": {"type": "string"}}},
        {"name": "knowledge/lookup", "description": "Lookup facts.", "parameters": {"entity": {"type": "string"}}}
    ]
    formatted_tools = "\n".join([f"- {tool['name']}: {tool['description']}" for tool in example_tools_list])

    # --- Selector Example ---
    formatted_selector_prompt = TOOL_SELECTOR_PROMPT_TEMPLATE.format(
        available_tools=formatted_tools,
        plan_step_thought=example_thought,
        plan_step_expected_outcome=example_outcome
    )
    print("--- Formatted Tool Selector Prompt Example ---")
    print(formatted_selector_prompt)
    expected_selector_output = '```json\n{\n  "reasoning": "Need current weather, use weather tool.",\n  "tool_name": "weather/get_current_weather"\n}\n```'
    print("\n--- Example Expected Selector Output ---")
    print(expected_selector_output.strip())

    # --- Input Generator Example ---
    selected_tool_name_inp = "weather/get_current_weather"
    selected_tool_desc_inp = "Get weather."
    selected_tool_params_inp = {"location": {"type": "string", "required": True}}
    example_thought_inp = "Need input for weather tool for London."
    example_outcome_inp = "Weather tool arguments for London."

    formatted_input_prompt = TOOL_INPUT_PROMPT_TEMPLATE.format(
        tool_name=selected_tool_name_inp,
        tool_description=selected_tool_desc_inp,
        tool_parameters_json=json.dumps(selected_tool_params_inp),
        plan_step_thought=example_thought_inp,
        plan_step_expected_outcome=example_outcome_inp
    )
    print("\n--- Formatted Tool Input Generator Prompt Example ---")
    print(formatted_input_prompt)
    expected_input_json_output = '```json\n{\n  "location": "London, UK"\n}\n```'
    print("\n--- Example Expected Input Generator Output ---")
    print(expected_input_json_output.strip())