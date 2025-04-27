# agent/solver.py
import json # Import json for _format_evidence
from typing import List, Dict, Any
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate

from .state import Evidence, ReWOOState

# Basic prompt template for the Solver
SOLVER_TEMPLATE = """
Given the original query and the collected evidence from plan execution, synthesize a final answer.
Base your answer ONLY on the provided evidence. Be comprehensive and directly answer the query.

Original Query: {query}

Collected Evidence:
{evidence_summary}

Final Answer:
"""

def _format_evidence(evidence_list: List[Evidence]) -> str:
    """ Formats the collected evidence for the solver prompt. """
    if not evidence_list:
        return "No evidence collected."

    formatted_evidence = []
    # Sort evidence by step index for logical flow
    sorted_evidence = sorted(evidence_list, key=lambda e: e['step_index'])

    for evidence in sorted_evidence:
        tool_result = evidence['tool_result']
        step_info = f"Evidence from Step {evidence['step_index']} (Tool: {tool_result['tool_name']}):"
        if tool_result.get('error'):
            result_str = f"  Error: {tool_result['error']}"
        else:
            # Try to format output nicely, fallback to string conversion
            try:
                output_str = json.dumps(tool_result['output'], indent=2) if isinstance(tool_result['output'], (dict, list)) else str(tool_result['output'])
                result_str = f"  Output: {output_str}"
            except Exception:
                result_str = f"  Output: {str(tool_result['output'])}" # Fallback

        # Include the processed evidence if available (might be more useful than raw output)
        processed_str = f"  Processed: {evidence.get('processed_evidence', 'N/A')}"

        formatted_evidence.append(f"{step_info}\n{result_str}\n{processed_str}")

    return "\\n---\\n".join(formatted_evidence)


async def generate_final_answer(
    llm: BaseLanguageModel,
    state: ReWOOState
) -> str:
    """
    Generates the final answer using the LLM based on the collected evidence.

    Args:
        llm: The language model instance to use for solving.
        state: The current ReWOO state containing the query and evidence.

    Returns:
        The generated final answer string.
    """
    query = state['input_query']
    evidence_list = state.get('evidence', []) # Get evidence, default to empty list

    evidence_summary = _format_evidence(evidence_list)
    prompt = PromptTemplate.from_template(SOLVER_TEMPLATE)
    solver_chain = prompt | llm

    print("\nGenerating final answer...")
    # Use async invocation if LLM supports it
    if hasattr(solver_chain, 'ainvoke'):
        final_answer_raw = await solver_chain.ainvoke({
            "query": query,
            "evidence_summary": evidence_summary
        })
    else:
        final_answer_raw = solver_chain.invoke({
            "query": query,
            "evidence_summary": evidence_summary
        })

    # Extract string content from potential AIMessage or similar object
    if isinstance(final_answer_raw, dict) and hasattr(final_answer_raw, 'content'):
        final_answer_str = final_answer_raw.content
    elif isinstance(final_answer_raw, str):
        final_answer_str = final_answer_raw
    else:
        print(f"Warning: Unexpected raw answer type: {type(final_answer_raw)}. Attempting to convert to string.")
        final_answer_str = str(final_answer_raw)

    print(f"\n--- Raw Final Answer ---\n{final_answer_str}\n------------------------")
    return final_answer_str.strip()

# Example Usage
if __name__ == "__main__":
    import asyncio
    from langchain_community.llms.fake import FakeListLLM

    async def run_solver_example():
        # --- Setup Fake LLM and Evidence for testing ---
        fake_answer = "Based on the search results (Evidence 1), the ReWOO pattern uses Planner, Worker, and Solver components. The calculated complexity (Evidence 2 & 3) is 12."
        fake_llm = FakeListLLM(responses=[fake_answer])

        example_state: ReWOOState = {
            "input_query": "Explain the ReWOO pattern and calculate its complexity based on components.",
            "plan": [], # Not needed for solver test itself
            "current_step_index": 3, # Example
            "evidence": [
                {
                    "step_index": 1,
                    "tool_result": {"tool_name": "web_search_placeholder/search", "output": "ReWOO uses Planner, Worker, Solver.", "error": None},
                    "processed_evidence": "ReWOO components are Planner, Worker, Solver."
                },
                {
                    "step_index": 2,
                    "tool_result": {"tool_name": "another_server/calculate", "output": 8, "error": None}, # 5 + 3
                    "processed_evidence": "Worker + Solver Complexity = 8"
                },
                 {
                    "step_index": 3,
                    "tool_result": {"tool_name": "another_server/calculate", "output": 12, "error": None}, # 8 + 4
                    "processed_evidence": "Total Complexity = 12"
                }
            ],
            "final_answer": None,
            "error": None
        }
        # --- ---

        try:
            print(f"Running solver for query: '{example_state['input_query']}'")
            final_answer = await generate_final_answer(fake_llm, example_state)

            print(f"\nGenerated Final Answer:\n{final_answer}")
            assert final_answer == fake_answer
            print("\nSolver example finished successfully.")

        except Exception as e:
            print(f"\nAn error occurred during solver example: {e}")

    asyncio.run(run_solver_example()) 