from langchain_core.prompts import PromptTemplate

FINAL_ANSWER_PROMPT_TEMPLATE = PromptTemplate.from_template(
    """Synthesize a final answer to the query using ONLY the provided evidence.
Respond ONLY with the final answer text. No other text.

Original Query: {original_query}

Collected Evidence:
{collected_evidence}

Instructions:
1. Base the answer *solely* on the Collected Evidence.
2. If evidence is insufficient or contradictory, state that clearly.
3. If evidence indicates a tool failure, report the failure.
4. Be concise and directly answer the Original Query.
5. Do NOT add external knowledge or repeat the query/evidence markers.

Final Answer:
"""
)

# Example Usage (for testing)
if __name__ == "__main__":
    import json

    example_query = "Compare the populations of Paris and London."
    example_evidence_list = [
        {"population_paris": 2100000},
        {"population_london": 9000000},
        "Error: Tool execution failed."
    ]

    # Format evidence simply
    formatted_evidence = "\n".join([
        f"- {json.dumps(ev) if isinstance(ev, dict) else str(ev)}"
        for ev in example_evidence_list
    ])

    formatted_prompt = FINAL_ANSWER_PROMPT_TEMPLATE.format(
        original_query=example_query,
        collected_evidence=formatted_evidence
    )

    print("--- Formatted Final Answer Prompt Example ---")
    print(formatted_prompt)

    expected_answer = "London (population approx. 9,000,000) has a larger population than Paris (population approx. 2,100,000). An error occurred during tool execution."
    print("\n--- Example Expected LLM Output ---")
    print(expected_answer)