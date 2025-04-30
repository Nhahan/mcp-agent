from langchain_core.prompts import PromptTemplate

# Simplified prompt for evidence extraction
EVIDENCE_PROMPT_TEMPLATE = PromptTemplate.from_template(
    """Extract the single most relevant piece of information from the tool output to meet the plan step goal.
Respond ONLY with a JSON object: {{"evidence": "<extracted_info_or_error>"}}. No other text.

Original Query: {original_query}
Plan Step Goal: {plan_step_thought} ({plan_step_expected_outcome})
Tool Used: {tool_name}
Tool Output:
{processed_tool_output}

Instructions:
1. Find the single most important fact in Tool Output relevant to the Plan Step Goal.
2. If the output shows an error or no relevant info, state that clearly as the evidence (e.g., "Tool execution failed.").
3. Output ONLY the JSON: {{"evidence": "<your_extracted_evidence_string>"}}.

Example Output (Success):
```json
{{
  "evidence": "The population of Paris is 2,141,000."
}}
```

Example Output (Error):
```json
{{
  "evidence": "Tool execution failed: Network error."
}}
```

Output the JSON evidence now.
"""
)

# Example Usage (for testing)
if __name__ == "__main__":
    import json

    example_query = "What is the population of Paris?"
    example_thought = "Look up Paris population."
    example_outcome = "Population number."
    example_tool = "knowledge/lookup"
    example_processed_output = {"population": 2141000, "country": "France"}

    formatted_prompt = EVIDENCE_PROMPT_TEMPLATE.format(
        original_query=example_query,
        plan_step_thought=example_thought,
        plan_step_expected_outcome=example_outcome,
        tool_name=example_tool,
        processed_tool_output=json.dumps(example_processed_output)
    )

    print("--- Formatted Evidence Generator Prompt Example ---")
    print(formatted_prompt)

    expected_json_output = '```json\n{\n  "evidence": "The population of Paris is 2,141,000."\n}\n```'
    print("\n--- Example Expected LLM Output ---")
    print(expected_json_output.strip())