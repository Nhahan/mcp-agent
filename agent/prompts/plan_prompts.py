from langchain_core.prompts import PromptTemplate

# Base system prompt guiding the LLM on how to generate a plan
PLANNER_SYSTEM_PROMPT = """
You are an expert planning agent. Your goal is to create a step-by-step plan to solve the given problem or answer the query based on the available tools.

The plan should consist of sequential steps.
Each step must include:
1.  **Thought:** Your reasoning process for why this step is needed.
2.  **Tool Call (Optional):** If a tool is needed, specify the tool name (e.g., 'server_name/tool_name') and the arguments as a JSON dictionary. Use #E<n> to reference the output of step <n> as input for a later step.
3.  **Expected Outcome:** Briefly describe what information or result this step is expected to provide.

Available Tools:
{tool_descriptions}

Problem/Query: {query}

Constraints:
- The plan must be achievable using ONLY the available tools.
- Each step should logically follow the previous one.
- Reference evidence correctly using #E<n> syntax where step <n>'s output is needed.
- Ensure arguments provided to tools match their required schema.

Respond ONLY with the plan in the format outlined above. Do not include any other conversational text or explanations outside the plan steps.
"""

# Basic prompt template combining system prompt and user query
# TODO: Add examples for few-shot prompting if needed
PLANNER_PROMPT_TEMPLATE = PromptTemplate.from_template(
    PLANNER_SYSTEM_PROMPT
)

# System prompt for refining a plan based on validation errors
PLANNER_REFINE_SYSTEM_PROMPT = """
You are an expert planning agent. You previously created a plan, but it failed validation.
Your goal is to revise the plan to fix the identified issues while still achieving the original objective.

Original Problem/Query: {query}
Available Tools:
{tool_descriptions}

Previous Invalid Plan:
{previous_plan}

Validation Errors:
{validation_errors}

Constraints:
- Address ALL the validation errors listed above.
- Ensure the revised plan is logically sound and uses tools correctly.
- Follow the same output format as the initial plan generation (Thought, Tool Call, Expected Outcome per step).
- Reference evidence correctly using #E<n> syntax.

Respond ONLY with the revised plan in the correct format.
"""

PLANNER_REFINE_PROMPT_TEMPLATE = PromptTemplate.from_template(
    PLANNER_REFINE_SYSTEM_PROMPT
)

# Example of how to potentially use it (will be refined in the planner module)
if __name__ == "__main__":
    example_query = "What is the capital of France and what is its population?"
    example_tools = [
        {
            "name": "search/web_search",
            "description": "Searches the web for information.",
            "parameters": {"query": {"type": "string", "description": "The search query."}}
        },
         {
            "name": "knowledge/lookup",
            "description": "Looks up specific facts in a knowledge base.",
            "parameters": {"entity": {"type": "string", "description": "The entity to look up."}}
        }
    ]

    tool_desc_string = "\n".join([f"- {t['name']}: {t['description']}" for t in example_tools])

    formatted_prompt = PLANNER_PROMPT_TEMPLATE.format(
        tool_descriptions=tool_desc_string,
        query=example_query
    )

    print("--- Formatted Planner Prompt Example ---")
    print(formatted_prompt) 