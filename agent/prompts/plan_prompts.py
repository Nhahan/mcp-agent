from typing import List
from langchain_core.prompts import PromptTemplate

# Simplified prompt for initial plan generation - YAML output
PLANNER_PROMPT_TEMPLATE = PromptTemplate.from_template(
    """Create a step-by-step plan to solve the query using the available tools.
Respond ONLY with a plan in the YAML format specified below. No other text before or after the plan block.

Available Tools:
{tool_descriptions} # <-- Use the tool names EXACTLY as provided in this list.

**CRITICAL INSTRUCTIONS for Planning:**
1.  **Analyze Query:** Understand the user's request (`{query}`). Extract key details needed for the overall goal.
2.  **Plan Agent Actions:** Break down the query into sequential steps. Each `plan` description should accurately reflect the action needed.
3.  **Distinguish Action Types:** Decide if a step is internal reasoning (`tool_call: null`) or requires an external tool.
4.  **Check Tool Availability & Schema:** For each external action, find the corresponding tool in `Available Tools` whose description BEST MATCHES the `plan` description for that step. **You MUST use the tool name EXACTLY as it appears in the `Available Tools` list above.** **CRITICALLY EXAMINE its `Arguments Schema`** to identify the **REQUIRED argument names and types**.
5.  **Use Tool (If Available):**
    *   If a tool **EXISTS** in `Available Tools` and **MATCHES the plan step objective**:
        *   Use its **`name` copied EXACTLY and DIRECTLY from the `Available Tools` list**.
        *   Specify `evidence_variable` (e.g., `"#E1"`).
        *   Provide `arguments` as a **YAML MAPPING (indented key-value pairs)**, NOT as a string containing JSON.
        *   **FOLLOW SCHEMA STRICTLY:** Based on the tool's `description` and the current `plan` step's objective, determine the necessary information to pass. Provide this information using **ONLY the argument names defined in the tool's `Arguments Schema`**. Ensure the values match the required types and YAML formatting rules (strings quoted if needed, numbers/booleans unquoted). Obtain values from the original `query`, internal reasoning, or previous evidence variables (like `"#E1"`) as appropriate.
        *   **USE ONLY SCHEMA ARGUMENTS:** **YOU MUST USE *ONLY* THE ARGUMENT NAMES EXACTLY AS SPECIFIED in the tool's `Arguments Schema`**. DO NOT use arguments not defined in the schema. 
        *   **VALID YAML VALUES:** Ensure string values are properly quoted (usually only needed if they contain special YAML characters). Remove any extraneous characters (like trailing '#', comments) from the values.
        *   **IMPORTANT FOR CONTENT GENERATION:** If the user query asks for specific content (like a recipe, code, etc.) AND the selected tool's purpose is to create that content, you MUST generate the **FULL, DETAILED, and COMPLETE content** as the value for the appropriate argument (identified from the schema) within the `arguments` YAML mapping. **DO NOT use placeholders like '...' or brief summaries.**
        *   The YAML example below shows the **STRUCTURE**. **YOU MUST USE ACTUAL TOOL NAMES AND ARGUMENTS FROM THE SCHEMA.** **DO NOT include any comments (starting with #) inside the YAML output.**
6.  **Acknowledge Limitation (If No Tool):** If **NO TOOL** matches the plan step objective, state the limitation and set `tool_call: null`.
7.  **Reflect Query Details:** Map query details (like specific paths or values) to the **CORRECT arguments identified from the `Arguments Schema`**.
8.  **YAML Format:** Respond ONLY with the YAML plan adhering strictly to the format below. **NO COMMENTS (#) WITHIN THE YAML BLOCK.**

Query: {query}

YAML Output Format Example (Illustrative Structure ONLY - Use REAL tools/args based on `Available Tools` schema. **NO COMMENTS INSIDE**):
```yaml
steps:
  - plan: "Internal reasoning step."
    tool_call: null
  - plan: "Call an external tool."
    tool_call:
       evidence_variable: "#E1"
       tool_name: "tool_name_from_list" # Use the actual tool name
       # IMPORTANT: arguments is a YAML MAPPING, not a string
       arguments:
         schema_arg_name_1: "value_from_query_or_reasoning" # Indented key-value pairs
         schema_arg_name_2: true                          # Values formatted as YAML types
         content_arg_if_applicable: |                     # Example for multi-line content
           Generated content here.
           Example newline.
  - plan: "Another step, potentially using evidence."
    tool_call:
       evidence_variable: "#E2"
       tool_name: "another_tool_name_from_list"
       arguments:
         required_arg_from_this_schema: "#E1" # Referencing previous evidence
```

Output the plan in the specified YAML format now.
"""
)

# Simplified prompt for refining a plan - YAML output
PLANNER_REFINE_PROMPT_TEMPLATE = PromptTemplate.from_template(
    """Revise the plan based on the errors. Respond ONLY with a plan in the YAML format specified below. No other text.

Available Tools:
{tool_descriptions} # <-- Use the tool names EXACTLY as provided in this list.

**CRITICAL INSTRUCTIONS for Revising:**
1.  **Analyze Error & Query:** Understand the `Validation Errors` and the `Original Query`.
2.  **Address Errors:**
    *   **Invalid Tool:** RE-CHECK `Available Tools`, find the tool whose description **BEST MATCHES** the objective of the failed `plan` step, and use its **EXACT** correct name **as listed**. If none exists, `tool_call: null`.
    *   **Invalid Arguments Format:** If the error indicates `arguments` was not a mapping (dictionary), **FIX the YAML structure** so `arguments:` is followed by **correctly indented key-value pairs**. Do NOT output `arguments` as a single string containing JSON.
    *   **Mismatched/Missing Arguments:** **CONSULT the `Arguments Schema` for the specific tool in `Available Tools`**. Determine the correct information needed based on the `plan` step objective and the tool `description`. Provide this information using **ONLY the argument names defined in the schema** within the `arguments` YAML mapping. Get values from the `Original Query` or previous evidence (`#E...`) if applicable. Fix the keys/values in the mapping.
    *   **Insufficient Content:** If the error indicates content was missing or just placeholders, AND the user query required specific content creation, **RE-GENERATE the FULL and DETAILED content** as the value for the appropriate argument in the `arguments` YAML mapping for the relevant tool.
    *   **Logical Error:** Correct the plan logic/tool choices.
    *   The YAML example below shows the **STRUCTURE**. **YOU MUST USE ACTUAL TOOL NAMES AND ARGUMENTS FROM THE SCHEMA.** **DO NOT include any comments (starting with #) inside the YAML output.**
3.  **Distinguish Action Types:** Re-evaluate if steps need `tool_call: null` or a tool.
4.  **Reflect Query Details:** Ensure query specifics map to the **CORRECT arguments identified from the tool's `Arguments Schema`** within the YAML mapping.
5.  **Maintain Structure:** Follow YAML formatting rules.
6.  **YAML Format:** Respond ONLY with the revised YAML plan. **NO COMMENTS (#) WITHIN THE YAML BLOCK.**

Original Query: {query}

Previous Plan Attempt (YAML):
{previous_plan}

Validation Errors (from parsing or execution):
{validation_errors}

Example Output Format (Illustrative Structure ONLY - Use REAL tools/args based on `Available Tools` schema. **NO COMMENTS INSIDE**):
```yaml
steps:
  - plan: "Internal step."
    tool_call: null
  - plan: "Revised: Call external tool with corrected arguments and potentially complete content."
    tool_call:
       evidence_variable: "#E1"
       tool_name: "correct_tool_name_from_list" # Use the actual tool name
       # IMPORTANT: arguments is a YAML MAPPING
       arguments:
         schema_arg_name_1: "corrected_value_from_reasoning_or_query"
         content_arg_if_needed: | # Example multi-line content
           Revised full content here.
           Example newline.
  - plan: "Revised: Next step using correct arguments."
    tool_call:
       evidence_variable: "#E2"
       tool_name: "another_tool_name_from_list"
       arguments:
         required_arg_from_this_schema: "#E1"
```

Output the revised plan in the specified YAML format now.
"""
)

# Keep example usage for testing if needed
if __name__ == "__main__":
    import json
    import yaml # Added for example parsing

    example_query = "What is the capital of France and what is its population?"
    example_tools_desc = """- search/web_search: Searches the web.
  Arguments Schema (JSON): {"properties": {"query": {"type": "string"}}, "required": ["query"]}
- knowledge/lookup: Looks up facts.
  Arguments Schema (JSON): {"properties": {"entity": {"type": "string"}}, "required": ["entity"]}"""


    formatted_prompt = PLANNER_PROMPT_TEMPLATE.format(
        tool_descriptions=example_tools_desc,
        query=example_query
    )
    print("--- Formatted Planner Prompt Example (YAML Mapping Args) ---")
    print(formatted_prompt)

    # Example Previous Plan with Args as String
    example_previous_plan_yaml = """steps:
  - plan: "Find the capital of France."
    tool_call:
      evidence_variable: "#E1"
      tool_name: "search/web_search"
      arguments: '{"query": "capital France"}' # Args as string
"""
    example_errors = "Pydantic validation failed: arguments field expects a dict, got str"

    formatted_refine_prompt = PLANNER_REFINE_PROMPT_TEMPLATE.format(
        tool_descriptions=example_tools_desc,
        query=example_query,
        previous_plan=example_previous_plan_yaml,
        validation_errors=example_errors
    )
    print("\n--- Formatted Refine Planner Prompt Example (YAML Mapping Args) ---")
    print(formatted_refine_prompt)

    # Example of parsing the expected YAML output (Mapping)
    expected_yaml_output = """steps:
  - plan: "Find the capital of France using search."
    tool_call:
      evidence_variable: "#E1"
      tool_name: "search/web_search"
      arguments:
        query: "capital of France"
  - plan: "Look up the population of the capital found in #E1."
    tool_call:
      evidence_variable: "#E2"
      tool_name: "knowledge/lookup"
      arguments:
        entity: "#E1" """

    try:
        parsed_plan = yaml.safe_load(expected_yaml_output)
        print("\n--- Example Parsed YAML Output (Mapping Args) ---")
        print(json.dumps(parsed_plan, indent=2))
        assert isinstance(parsed_plan['steps'][0]['tool_call']['arguments'], dict)
        assert isinstance(parsed_plan['steps'][1]['tool_call']['arguments'], dict)
        print("Parsing test successful!")
    except yaml.YAMLError as e:
        print(f"\nError parsing example YAML: {e}")
