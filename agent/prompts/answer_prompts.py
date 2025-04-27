from langchain_core.prompts import PromptTemplate

# System prompt for the Final Answer Generator Agent
# This prompt instructs the LLM on how to synthesize the final answer
# based on the original query and collected evidence.
FINAL_ANSWER_SYSTEM_PROMPT = """
You are an expert Answer Synthesizer agent. Your goal is to generate a comprehensive and accurate final answer to the original query based solely on the provided evidence.

Original Query: {original_query}

Collected Evidence:
{collected_evidence}

Instructions:
1. Carefully review all pieces of collected evidence. Evidence might be direct tool outputs or summaries.
2. Synthesize the information from the evidence to directly answer the original query.
3. Ensure the final answer is factually accurate based *only* on the provided evidence. Do not add external knowledge or assumptions.
4. If the evidence is contradictory or insufficient to fully answer the query, state that limitation clearly in the answer.
5. If the evidence indicates a failure occurred (e.g., "Tool execution failed", "Error: Tool not found"), report that failure clearly as part of the answer or as the entire answer if appropriate.
6. The answer should be well-structured, coherent, and easy to understand.
7. Do not include evidence markers (like "Evidence from Step 1:") or repetition of the query in the final answer unless essential for clarity.
8. Respond ONLY with the final answer text.

Example Request 1:
Original Query: What is the capital of France and its population?
Collected Evidence:
- Evidence from Step 1: Paris is the capital of France.
- Evidence from Step 2: {{"population": 2141000, "source": "wiki_lookup"}}

Example Final Answer 1:
The capital of France is Paris, and its population is approximately 2,141,000 based on wiki_lookup.

Example Request 2:
Original Query: What is the weather like in London?
Collected Evidence:
- Evidence from Step 1: Error executing tool 'get_weather': Service unavailable.

Example Final Answer 2:
I could not retrieve the weather information for London because the execution of the 'get_weather' tool failed due to the service being unavailable.

Example Request 3:
Original Query: Who directed the movie Inception?
Collected Evidence:
- Evidence from Step 1: {{"title": "Inception", "genre": "Sci-Fi Action", "starring": ["Leonardo DiCaprio"]}}
- Evidence from Step 2: Error: Tool 'find_director' not found.

Example Final Answer 3:
Based on the available evidence, Inception is a Sci-Fi Action film starring Leonardo DiCaprio. However, I could not determine the director because the 'find_director' tool was not found.

Final Answer:
"""

# Prompt template for Final Answer Generation
FINAL_ANSWER_PROMPT_TEMPLATE = PromptTemplate.from_template(
    FINAL_ANSWER_SYSTEM_PROMPT
)

# Example Usage (for testing and demonstration)
if __name__ == "__main__":
    import json # Ensure json is imported for the example

    example_query = "Compare the populations of Paris and London."
    # Example evidence list (could be strings, dicts, etc.)
    example_evidence_list = [
        "Paris is the capital of France.",
        {"population_paris": 2100000},
        "London is the capital of the UK.",
        {"population_london": 9000000},
        "Error executing tool 'get_comparison_details': Timeout."
    ]

    # Format evidence for the prompt (simple string representation for demo)
    formatted_evidence = "\n".join([
        f"- Evidence from Step {i+1}: {json.dumps(ev) if isinstance(ev, dict) else str(ev)}"
        for i, ev in enumerate(example_evidence_list)
    ])

    # Format the complete prompt
    formatted_prompt = FINAL_ANSWER_PROMPT_TEMPLATE.format(
        original_query=example_query,
        collected_evidence=formatted_evidence
    )

    print("--- Formatted Final Answer Prompt Example ---")
    print(formatted_prompt)

    # Example expected final answer text from the LLM:
    expected_answer = "London (population approx. 9,000,000) has a significantly larger population than Paris (population approx. 2,100,000). I could not retrieve further comparison details due to a tool execution timeout."
    print("\n--- Example Expected LLM Output (Final Answer Text) ---")
    print(expected_answer) 