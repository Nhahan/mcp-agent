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

# Updated PLANNER_PROMPT_TEMPLATE focusing on standard JSON output
PLANNER_PROMPT_TEMPLATE = PromptTemplate.from_template(
    """Your task is to create a step-by-step plan to solve the given query using the available tools. 
    Respond **ONLY** with a valid JSON object enclosed in a single ```json ... ``` block. 
    Do **NOT** include any other text, comments, or explanations before or after the JSON block.

    Available Tools (Use EXACT names):
    {tool_descriptions}

    Query: {query}

    Required JSON Output Schema:
    {{
      "steps": [ // MUST be a list of step objects
        {{ // Example Step using a tool
          "plan": "<Your reasoning for this step>",
          "tool_call": {{ // Object if using a tool
            "evidence_variable": "#E<n>",
            "tool_name": "<tool_name_from_available_tools>",
            "arguments": {{ "<arg_name>": "<arg_value>", ... }} // Arguments as a JSON object
          }}
        }}, // <-- IMPORTANT: Comma needed between step objects in the list!
        {{ // Example Step *without* using a tool (providing a known fact)
          \"plan\": \"The capital of France is Paris.\", // Actual factual content in 'plan'
          \"tool_call\": null // <-- MUST be null if no tool is needed
        }}
        // ... more steps possible, remember the comma between them!
      ]
    }}

    Constraints & Instructions:
    - The JSON object MUST contain a single key "steps" which is a list of step objects.
    - **Each step object in the "steps" list MUST be separated by a comma (,) except for the last one.**
    - **Inside each object, key-value pairs MUST be separated by a comma (,) except for the last pair.**
    - Each step object MUST contain a "plan" key with your reasoning as a string.
    - Each step object MUST contain a "tool_call" key.
    - **Query Relevance:** Critically evaluate if a tool is genuinely necessary and **relevant** to answer the specific 'Query: {query}'. Do **NOT** use tools that are unrelated (e.g., time tools for a cooking query).
    - **Prioritize `null`:** If the step involves reasoning, summarizing, or can be answered from general knowledge without external data, strongly prefer setting `tool_call` to `null`.
    - **Content for `null` tool_call:** If `tool_call` is `null` because the step can be answered using general knowledge (like providing instructions, definitions, or summaries), the "plan" field for that step **MUST contain the actual instructional text or information** that contributes directly to the final answer. Do not just describe *what* you will do (e.g., "List the ingredients"); instead, *provide* the information (e.g., "The ingredients are: flour, sugar, eggs...").
    - If a tool is used, "tool_call" MUST be an object containing:
        - \"evidence_variable\": A string like \"#E1\", \"#E2\", etc., incrementing for each tool call step.
        - \"tool_name\": The exact name of the tool from the 'Available Tools' list.
        - \"arguments\": **MUST be a valid JSON object**. If a tool takes no arguments, use an empty JSON object: `{{}}`. **Do NOT use the string `\"{{}}\"`**. Ensure keys and string values within the arguments object are enclosed in double quotes.
            - **Argument Accuracy:** You MUST provide accurate and necessary arguments based on the tool's description and the query context. Do not guess arguments. If required arguments are missing in the query, set `tool_call` to `null` or use a tool to find the missing information first.
    - If **NO external tool is required** for a step (e.g., simple reasoning, summarizing information already gathered, providing instructions based on general knowledge like preheating an oven), you **MUST set the value of \"tool_call\" directly to `null`**. Do **NOT** create a `tool_call` object with null/empty values inside in this case.
    - **Crucially, ensure the entire output is a single, valid JSON object.** Pay EXTREME attention to quotes (\"), commas (,), braces ({{}}), and brackets ([]).
    - Extract necessary entities from the 'Query: {query}' to use as arguments in the "arguments" object.
    - The plan should logically progress towards answering the query.

    Begin! Output the JSON plan now.
    """
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

# Updated PLANNER_REFINE_PROMPT_TEMPLATE for standard JSON output
PLANNER_REFINE_PROMPT_TEMPLATE = PromptTemplate.from_template(
    """The previous attempt to create a plan for the task failed parsing or validation. 
    You MUST generate a revised plan adhering to the ORIGINAL query and the JSON schema provided below.
    Respond **ONLY** with a single, valid JSON object enclosed in a ```json ... ``` block. 
    Do **NOT** include any other text, comments, or explanations.

    Available Tools (Use EXACT names):
    {tool_descriptions}

    Original Query: {query}

    Previous Plan Attempt (May contain errors or be incomplete):
    ------
    {previous_plan} 
    ------

    Validation/Parsing Errors from Previous Attempt:
    ------
    {validation_errors}
    ------

    Required JSON Output Schema:
    {{
      "steps": [ // MUST be a list of step objects
        {{
          "plan": "<Your reasoning for this step>",
          "tool_call": {{ // Object if using a tool
            "evidence_variable": "#E<n>", 
            "tool_name": "<tool_name_from_available_tools>",
            "arguments": {{ "<arg_name>": "<arg_value>", ... }} // Arguments as a JSON object
          }}
        }}, // <-- IMPORTANT: Comma needed between step objects!
        {{
          "plan": "<Reasoning for the final answer construction>",
          "tool_call": null // null for the final step
        }}
        // ... more steps possible, remember the comma between them!
      ]
    }}

    Revised Plan Instructions:
    - Carefully analyze the 'Original Query', 'Previous Plan Attempt', and 'Validation/Parsing Errors'.
    - Generate a COMPLETELY NEW plan in the specified JSON format that addresses the errors and solves the original query.
    - **Correct Tool Usage:** Pay special attention to selecting tools that are **directly relevant** to the 'Original Query'. Avoid using unrelated tools even if they appear in the available list. Ensure **all required arguments** for chosen tools are provided correctly in the `arguments` JSON object.
    - **Prioritize `null`:** If a step in the revised plan involves reasoning or summarizing without needing external data, prefer setting `tool_call` to `null`.
    - **Ensure the new plan follows all JSON syntax rules precisely:**
        - **Commas (,) are REQUIRED between elements in the \"steps\" list (e.g., `}}, {{`).**
        - **Commas (,) are REQUIRED between key-value pairs within all objects (e.g., `"key1": "value1", "key2": "value2"`).**
        - All keys and string values MUST be enclosed in double quotes (\").
        - Ensure all brackets ([]) and braces ({{}}) are correctly matched and placed.
    - Follow all constraints mentioned in the initial planning instructions (correct tool names, arguments as JSON objects, `null` for final step tool_call, etc.).
    - **DOUBLE-CHECK the final JSON for syntax errors, especially missing commas, before outputting.**

    Begin! Output the revised JSON plan now.
    """
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