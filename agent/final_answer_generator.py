# agent/final_answer_generator.py
import json
import re
from typing import Dict, Any, Optional, List

from langchain_core.language_models import BaseLanguageModel

# Assuming Evidence type is defined or just using string for now
# from .state import Evidence
Evidence = str # Placeholder type

# Import the dedicated prompt template
from .prompts.final_answer_prompts import FINAL_ANSWER_PROMPT_TEMPLATE

class FinalAnswerGenerationError(Exception):
    """Custom exception for final answer generation errors."""
    pass

class FinalAnswerGenerator:
    """Generates the final answer by synthesizing collected evidence using an LLM."""

    def __init__(
        self,
        llm: BaseLanguageModel,
        max_retries: int = 1
    ):
        """
        Initializes the FinalAnswerGenerator.

        Args:
            llm: The language model instance for generating the final answer.
            max_retries: Maximum attempts if generation/parsing fails.
        """
        self.llm = llm
        self.max_retries = max_retries
        self.prompt_template = FINAL_ANSWER_PROMPT_TEMPLATE # Use imported template

    def _format_evidence_for_prompt(self, evidence_list: List[Evidence]) -> str:
        """Formats the list of evidence strings into a single block for the prompt."""
        if not evidence_list:
            return "No evidence collected."
        
        formatted = "\n".join([
            f"- Evidence {i+1}: {evidence}"
            for i, evidence in enumerate(evidence_list)
        ])
        return formatted

    def _parse_final_answer_response(self, response_content: str) -> str:
        """Parses the LLM response to extract the final answer string."""
        # Assuming the LLM might respond with just the answer text, or maybe wrapped
        # For now, let's assume it returns the text directly or within a simple structure
        # This might need refinement based on the actual prompt design (Task 9.2)
        try:
            # Option 1: Check for a specific JSON structure if the prompt demands it
            # Example: {"final_answer": "..."}
            # match = re.search(r'```json\n({.*?})\n```', response_content, re.DOTALL)
            # if match:
            #     parsed_json = json.loads(match.group(1))
            #     if 'final_answer' in parsed_json and isinstance(parsed_json['final_answer'], str):
            #         return parsed_json['final_answer'].strip()
            #     else:
            #         raise ValueError("JSON response missing 'final_answer' string.")
            
            # Option 2: Assume the response IS the final answer string (maybe cleanup needed)
            # Simple cleanup: remove potential markdown code fences if present
            cleaned_response = re.sub(r'^```(?:json|text)?\n', '', response_content.strip(), flags=re.IGNORECASE)
            cleaned_response = re.sub(r'\n```$', '', cleaned_response)
            
            if not cleaned_response:
                 raise ValueError("LLM response for final answer was empty after cleaning.")

            return cleaned_response.strip() 

        except Exception as e:
            # Catch JSON errors if Option 1 is used, or any other unexpected error
            raise FinalAnswerGenerationError(f"Failed to parse final answer response: {e}\nResponse: {response_content}") from e

    async def generate_final_answer(
        self,
        original_query: str,
        evidence_list: List[Evidence]
        # Add plan context if needed by the prompt
    ) -> str:
        """
        Generates the final answer based on the original query and collected evidence.

        Args:
            original_query: The initial query to the agent.
            evidence_list: A list of evidence strings collected from tool executions.

        Returns:
            A string containing the generated final answer.

        Raises:
            FinalAnswerGenerationError: If the final answer cannot be generated.
        """
        last_error: Optional[Exception] = None
        formatted_evidence = self._format_evidence_for_prompt(evidence_list)

        # Use the prompt template
        prompt_input = {
            "original_query": original_query,
            "collected_evidence": formatted_evidence
        }
        generation_chain = self.prompt_template | self.llm

        for attempt in range(self.max_retries + 1):
            print(f"\n--- Final Answer Generation Attempt {attempt + 1}/{self.max_retries + 1} ---")
            try:
                # Invoke LLM using the chain
                print(f"Invoking LLM for final answer...")
                llm_response = await generation_chain.ainvoke(prompt_input)

                # Extract content
                if hasattr(llm_response, 'content'):
                    response_content = llm_response.content
                elif isinstance(llm_response, str):
                    response_content = llm_response
                else:
                    print(f"Warning: Unexpected LLM response type for final answer: {type(llm_response)}. Converting to string.")
                    response_content = str(llm_response)

                print(f"Raw LLM Response:\n{response_content}") # Debug output

                # Parse response
                final_answer = self._parse_final_answer_response(response_content)
                print(f"Generated Final Answer: {final_answer}")

                # TODO: Add validation for answer quality/relevance (Task 9.3)

                return final_answer # Return successfully generated answer

            except FinalAnswerGenerationError as e:
                print(f"Attempt {attempt + 1} Error: {e}")
                last_error = e
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Attempt {attempt + 1} Unexpected Error: {e}")
                last_error = FinalAnswerGenerationError(f"Unexpected error during final answer generation: {e}") from e

            # Retry logic
            if attempt < self.max_retries:
                print("Retrying final answer generation...")
            else:
                print("Max retries reached.")

        # If loop finishes
        raise FinalAnswerGenerationError("Failed to generate final answer after maximum retries.") from last_error

# Example Usage (requires async context and mocks)
if __name__ == "__main__":
    import asyncio
    from langchain_community.llms.fake import FakeListLLM

    example_query = "Compare the populations of Paris and London."
    example_evidence = [
        "The population of Paris is approximately 2.1 million.",
        "London has a population of around 9 million.",
        "Both are major European capital cities."
    ]

    async def run_final_answer_example():
        # Fake LLM response - just the final text in this example
        fake_llm_response = "London has a significantly larger population (around 9 million) compared to Paris (approximately 2.1 million). Both are major European capitals."
        
        # Test successful generation
        print("--- Testing Successful Final Answer Generation ---")
        fake_llm_success = FakeListLLM(responses=[fake_llm_response])
        generator_success = FinalAnswerGenerator(llm=fake_llm_success)
        try:
            answer = await generator_success.generate_final_answer(
                example_query, example_evidence
            )
            print(f"Success! Final Answer: {answer}")
            assert "London" in answer and "Paris" in answer and "population" in answer
        except FinalAnswerGenerationError as e:
            print(f"Error during successful generation test: {e}")

        # Test parsing failure (e.g., empty response) with retry
        print("\n--- Testing Final Answer Generation Failure (Parsing) with Retry ---")
        malformed_response = ""
        fake_llm_retry = FakeListLLM(responses=[malformed_response, fake_llm_response])
        generator_retry = FinalAnswerGenerator(llm=fake_llm_retry, max_retries=1)
        try:
            answer = await generator_retry.generate_final_answer(
                 example_query, example_evidence
            )
            print(f"Retry Success! Final Answer: {answer}")
            assert "London" in answer and "Paris" in answer
        except FinalAnswerGenerationError as e:
            print(f"Error during parsing retry test: {e}")

    asyncio.run(run_final_answer_example()) 