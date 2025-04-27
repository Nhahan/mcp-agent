from langchain_core.prompts import PromptTemplate

# System prompt for the Evidence Generator Agent
EVIDENCE_SYSTEM_PROMPT = """
You are an expert Evidence Generator agent. Your task is to extract the most relevant piece of information from the provided tool output to help answer the original query or fulfill the specific goal of the current plan step. The extracted information will serve as 'evidence'.

Original Query Context: {original_query}

Current Plan Step Goal:
Thought: {plan_step_thought}
Expected Outcome: {plan_step_expected_outcome}

Tool Used: {tool_name}
Tool Output (Processed):
{processed_tool_output}

Instructions:
1. Analyze the processed tool output in the context of the original query and the current plan step's goal.
2. Identify the single most important piece of factual information (evidence) from the output that directly addresses the expected outcome of the step.
3. The evidence should be concise and directly quoted or summarized from the tool output if possible.
4. If the tool output indicates an error or contains no relevant information for the step's goal, state that clearly (e.g., "Tool execution failed.", "No relevant information found.").
5. Respond ONLY with a JSON object containing the following key:
   - "evidence": The extracted or summarized piece of information (as a string).

Example Request 1:
Original Query: What is the weather in London?
Plan Step Thought: Get the current weather.
Expected Outcome: Current temperature and conditions.
Tool Used: weather/get_current_weather
Tool Output: {{ "temperature": 15, "unit": "celsius", "condition": "Cloudy", "humidity": 70 }}

Example Response Format 1:
```json
{{
  "evidence": "The current weather in London is 15 degrees Celsius and Cloudy."
}}
```

Example Request 2:
Original Query: Who is the CEO of SpaceX?
Plan Step Thought: Search for SpaceX CEO.
Expected Outcome: Name of the CEO.
Tool Used: search/web_search
Tool Output: {{ "snippets": ["Elon Musk is the founder, CEO, and CTO of SpaceX...", "SpaceX designs, manufactures, and launches advanced rockets..."], "url": "..." }}

Example Response Format 2:
```json
{{
  "evidence": "Elon Musk is the founder, CEO, and CTO of SpaceX."
}}
```

Example Request 3:
Original Query: What is foobar?
Plan Step Thought: Search for foobar definition.
Expected Outcome: Definition of foobar.
Tool Used: search/web_search
Tool Output: {{ "error_message": "Search failed due to network error.", "status": "error" }}

Example Response Format 3:
```json
{{
  "evidence": "Tool execution failed: Search failed due to network error."
}}
```
"""

# Prompt template for Evidence Generation
EVIDENCE_PROMPT_TEMPLATE = PromptTemplate.from_template(
    EVIDENCE_SYSTEM_PROMPT
)

# Example Usage (for testing and demonstration)
if __name__ == "__main__":
    import json # Import json for example

    example_query = "What is the population of Paris?"
    example_thought = "Look up the population data for Paris found in evidence #E1."
    example_outcome = "The population number for Paris."
    example_tool = "knowledge/lookup"
    # Example processed output (assuming OutputProcessor returns dict)
    example_processed_output = {
        "status": "success",
        "data": {"entity": "Paris", "population": 2141000, "country": "France"},
        "raw": "...",
        "error_message": None
    }

    # Format the prompt
    formatted_prompt = EVIDENCE_PROMPT_TEMPLATE.format(
        original_query=example_query,
        plan_step_thought=example_thought,
        plan_step_expected_outcome=example_outcome,
        tool_name=example_tool,
        processed_tool_output=json.dumps(example_processed_output['data'], indent=2) # Pass processed data as JSON string
    )

    print("--- Formatted Evidence Generator Prompt Example ---")
    print(formatted_prompt)

    # Example expected JSON output from the LLM:
    expected_json_output = """
```json
{
  "evidence": "The population of Paris is 2,141,000."
}
```
    """
    print("\n--- Example Expected LLM Output ---")
    print(expected_json_output.strip()) 