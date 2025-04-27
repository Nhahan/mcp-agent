# agent/evidence_generator.py
import json
import re
from typing import Dict, Any, Optional

from langchain_core.language_models import BaseLanguageModel

from .state import PlanStep # Assuming PlanStep is defined
from .prompts.evidence_prompts import EVIDENCE_PROMPT_TEMPLATE

class EvidenceGenerationError(Exception):
    """Custom exception for evidence generation errors."""
    pass

class EvidenceGenerator:
    """Generates concise evidence from processed tool outputs using an LLM."""

    def __init__(
        self,
        llm: BaseLanguageModel,
        max_retries: int = 1
    ):
        """
        Initializes the EvidenceGenerator.

        Args:
            llm: The language model instance for generating evidence.
            max_retries: Maximum attempts to retry generation if parsing fails.
        """
        self.llm = llm
        self.max_retries = max_retries
        self.prompt_template = EVIDENCE_PROMPT_TEMPLATE

    def _parse_evidence_response(self, response_content: str) -> str:
        """Parses the LLM response (expected JSON) to extract the evidence string."""
        try:
            # Extract JSON from potential markdown code blocks
            match = re.search(r'```json\n({.*?})\n```', response_content, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                json_str = response_content.strip()

            parsed_json = json.loads(json_str)

            if not isinstance(parsed_json, dict) or 'evidence' not in parsed_json:
                raise ValueError("Response JSON missing required key ('evidence').")

            evidence = parsed_json['evidence']
            if not isinstance(evidence, str):
                # Attempt to convert non-string evidence, or raise error
                print(f"Warning: Received non-string evidence, converting: {evidence}")
                evidence = str(evidence)
                # Alternatively, raise ValueError("'evidence' key must contain a string.")

            return evidence.strip() # Return the extracted evidence string

        except json.JSONDecodeError as e:
            raise EvidenceGenerationError(f"Failed to parse LLM response as JSON: {e}\nResponse: {response_content}") from e
        except ValueError as e:
             raise EvidenceGenerationError(f"Invalid response format: {e}\nResponse: {response_content}") from e
        except Exception as e:
            raise EvidenceGenerationError(f"Unexpected error parsing evidence response: {e}\nResponse: {response_content}") from e

    async def generate_evidence(
        self,
        original_query: str,
        plan_step: PlanStep,
        processed_tool_output: Dict[str, Any], # Output from OutputProcessor
        tool_name: str
    ) -> str:
        """
        Generates evidence based on the tool output and context.

        Args:
            original_query: The initial query to the agent.
            plan_step: The plan step that triggered the tool execution.
            processed_tool_output: The processed output from the OutputProcessor.
            tool_name: The name of the tool that was executed.

        Returns:
            A string containing the generated evidence.

        Raises:
            EvidenceGenerationError: If evidence cannot be generated after retries.
        """
        last_error: Optional[Exception] = None

        # Prepare the output string for the prompt
        # Include status and error message if processing failed
        if processed_tool_output.get('status') == 'error':
            output_for_prompt = json.dumps({
                "status": "error",
                "error_message": processed_tool_output.get('error_message')
            }, indent=2)
        else:
             # Pass the processed data to the prompt
             output_for_prompt = json.dumps(processed_tool_output.get('data'), indent=2)


        for attempt in range(self.max_retries + 1):
            print(f"\n--- Evidence Generation Attempt {attempt + 1}/{self.max_retries + 1} ---")
            try:
                # Prepare prompt input
                prompt_input = {
                    "original_query": original_query,
                    "plan_step_thought": plan_step.thought or "(No thought)",
                    "plan_step_expected_outcome": plan_step.expected_outcome or "(No outcome specified)",
                    "tool_name": tool_name,
                    "processed_tool_output": output_for_prompt
                }
                generation_chain = self.prompt_template | self.llm

                # Invoke LLM
                llm_response = await generation_chain.ainvoke(prompt_input)

                # Extract content
                if hasattr(llm_response, 'content'):
                    response_content = llm_response.content
                elif isinstance(llm_response, str):
                    response_content = llm_response
                else:
                    print(f"Warning: Unexpected LLM response type for evidence generation: {type(llm_response)}. Converting to string.")
                    response_content = str(llm_response)

                print(f"Raw LLM Response:\n{response_content}") # Debug output

                # Parse response
                evidence = self._parse_evidence_response(response_content)
                print(f"Generated Evidence: {evidence}")

                # TODO: Add validation for evidence quality/relevance (Task 8.5)

                return evidence # Return successfully generated evidence

            except EvidenceGenerationError as e:
                print(f"Attempt {attempt + 1} Error: {e}")
                last_error = e
            except Exception as e:
                # 원래 traceback.print_exc() 대신 기본 에러 메시지만 출력
                print(f"Attempt {attempt + 1} Unexpected Error: {str(e)}")
                # from e 구문을 사용하지 않고 에러 객체 생성
                last_error = EvidenceGenerationError(f"Unexpected error during evidence generation: {str(e)}")

            # Retry logic
            if attempt < self.max_retries:
                print("Retrying evidence generation...")
            else:
                print("Max retries reached.")

        # If loop finishes
        raise EvidenceGenerationError("Failed to generate evidence after maximum retries.") from last_error

# Example Usage (requires async context and mocks)
if __name__ == "__main__":
    import asyncio
    from langchain_community.llms.fake import FakeListLLM

    # Mock Plan Step
    mock_plan_step = PlanStep(
        step_index=0,
        thought="Find the capital of France.",
        expected_outcome="The capital city name.",
        status="pending"
    )
    original_query = "What is the capital of France?"
    tool_name = "search/web_search"
    # Example processed output from OutputProcessor
    processed_output = {
        "status": "success",
        "data": {"result": "Paris is the capital of France."},
        "raw": "...",
        "error_message": None
    }
    processed_output_error = {
        "status": "error",
        "data": None,
        "raw": "Timeout",
        "error_message": "Tool execution failed: Timeout"
    }

    async def run_evidence_example():
        # Fake LLM response matching the expected JSON format
        fake_llm_response_evidence = '''
```json
{
  "evidence": "Paris is the capital of France."
}
```
        '''
        fake_llm_response_error_evidence = '''
```json
{
  "evidence": "Tool execution failed: Timeout"
}
```
        '''

        # Test successful generation
        print("--- Testing Successful Evidence Generation ---")
        fake_llm_success = FakeListLLM(responses=[fake_llm_response_evidence])
        generator_success = EvidenceGenerator(llm=fake_llm_success)
        try:
            evidence = await generator_success.generate_evidence(
                original_query, mock_plan_step, processed_output, tool_name
            )
            print(f"Success! Generated Evidence: {evidence}")
            assert evidence == "Paris is the capital of France."
        except EvidenceGenerationError as e:
            print(f"Error during successful generation test: {e}")

        # Test generation when tool output was an error
        print("\n--- Testing Evidence Generation from Error Output ---")
        fake_llm_error = FakeListLLM(responses=[fake_llm_response_error_evidence])
        generator_error = EvidenceGenerator(llm=fake_llm_error)
        try:
            evidence = await generator_error.generate_evidence(
                original_query, mock_plan_step, processed_output_error, tool_name
            )
            print(f"Success! Generated Evidence from Error: {evidence}")
            assert "Tool execution failed" in evidence
        except EvidenceGenerationError as e:
            print(f"Error during error evidence generation test: {e}")

        # Test parsing failure with retry
        print("\n--- Testing Evidence Generation Failure (Parsing) with Retry ---")
        malformed_response = """```json\n{\n  "wrong_key": "Paris is the capital"\n}\n```"""
        fake_llm_retry = FakeListLLM(responses=[malformed_response, fake_llm_response_evidence])
        generator_retry = EvidenceGenerator(llm=fake_llm_retry, max_retries=1)
        try:
            evidence = await generator_retry.generate_evidence(
                original_query, mock_plan_step, processed_output, tool_name
            )
            print(f"Retry Success! Generated Evidence: {evidence}")
            assert evidence == "Paris is the capital of France."
        except EvidenceGenerationError as e:
            print(f"Error during parsing retry test: {e}")


    asyncio.run(run_evidence_example()) 