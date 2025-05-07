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
        *   Use its **`name` copied EXACTLY and DIRECTLY from the `Available Tools` list** as `tool_name`.
        *   Specify `evidence_variable` (e.g., `"E1"` - **IMPORTANT: No leading '#' in the YAML value for `evidence_variable`, just the letter and number like "E1", "E2" etc.**). The '#' prefix is an internal convention for referencing, not part of the YAML value itself.
        *   Provide `arguments` as a **YAML MAPPING (indented key-value pairs)**, NOT as a string containing JSON.
        *   **FOLLOW SCHEMA STRICTLY:** Based on the tool's `description` and the current `plan` step's objective, determine the necessary information to pass. Provide this information using **ONLY the argument names defined in the tool's `Arguments Schema`**. Ensure the values match the required types and YAML formatting rules (strings quoted if needed, numbers/booleans unquoted). Obtain values from the original `query`, internal reasoning, or previous evidence variables (e.g., `"#E1"` - when referencing evidence in an argument value, the '#' IS used).
        *   **USE ONLY SCHEMA ARGUMENTS:** **YOU MUST USE *ONLY* THE ARGUMENT NAMES EXACTLY AS SPECIFIED in the tool\'s `Arguments Schema`**. DO NOT use arguments not defined in the schema.
        *   **VALID YAML VALUES:** Ensure string values are properly quoted (usually only needed if they contain special YAML characters). Remove any extraneous characters (like trailing \'#\', comments) from the values.
        *   **IMPORTANT FOR CONTENT GENERATION:** If the user query asks for specific content (like a recipe, code, etc.) AND the selected tool's purpose is to create that content, you MUST generate the **FULL, DETAILED, and COMPLETE content** as the value for the appropriate argument (identified from the schema) within the `arguments` YAML mapping. **DO NOT use placeholders like \'...\' or brief summaries.**
        *   The YAML example below shows the **STRUCTURE**. **YOU MUST USE ACTUAL TOOL NAMES AND ARGUMENTS FROM THE SCHEMA.** **DO NOT include any comments (starting with #) inside the YAML output.**
6.  **Specify Evidence Source (Based on Tool Purpose, Description, and Input Parameters):**
    *   Analyze the tool's `description`, its `Arguments Schema`, and **critically review any `Parameters explained` sections or detailed argument descriptions.** What is the primary *output* of the tool versus what is a significant *input that needs to be recorded as the primary outcome* of this step?
    *   **Standard Output Tools:** If the tool's description clearly indicates it **returns the primary valuable data** (e.g., search results from a search tool, file content from a file reader, a list from a lister tool), then its direct output IS the evidence. In these cases, **OMIT** `evidence_input_key` or set it to `null`.
        Example: A `file_search` tool returns a list of files. The list itself is the evidence.
    *   **Input as Evidence Tools (e.g., Logging, Thinking, Context-Setting Tools):**
        If the tool's primary purpose is to **process, log, record, or stage a specific input argument for later use or as the main result of this thinking/logging step**, and the tool itself might return only metadata, a status, or an ID, then the **value of that key input argument SHOULD be the evidence.**
        **CAREFULLY EXAMINE the `Arguments Schema` and `Parameters explained` section of the tool description.** 
        Look for parameters that represent the core content. For example, for a tool like `sequentialthinking` (with `thought` parameter), you **MUST set `evidence_input_key: "thought"`.**
        If such a parameter exists and it clearly holds the substantive information, **YOU MUST set `evidence_input_key` to that parameter's exact name.**
    *   **Decision Criteria:** If the tool's direct return value is merely status, metadata, or an ID, and a specific *input argument* holds the actual content/thought/data this step is designed to produce or record as its main outcome, **YOU MUST use `evidence_input_key` and set it to the name of that input argument.**
    *   If unclear, or if the tool's direct output genuinely seems to be the most valuable, omit `evidence_input_key`.

Output Format Example:
```yaml
steps:
  - plan: "Briefly think about the user's main goal."
    tool_call: null # Internal thought, no tool needed yet.
  - plan: "Search for recent news about AI advancements."
    tool_call:
      evidence_variable: "E1"
      tool_name: "web_search"
      arguments:
        query: "recent AI advancements"
      # evidence_input_key is omitted or null: web_search output (search results) is the evidence.
  - plan: "Log the first step of a detailed thought process about the search results."
    tool_call:
      evidence_variable: "E2"
      tool_name: "sequentialthinking" # Actual tool name
      arguments:
        thought: "The search results indicate significant progress in large language models." # Use actual param name
        nextThoughtNeeded: true
        thoughtNumber: 1
        totalThoughts: 3 # Example, LLM should determine this
      evidence_input_key: "thought" # The 'thought' input is the actual evidence for sequentialthinking.
  - plan: "Record the user's specific action for auditing."
    tool_call:
      evidence_variable: "E3"
      tool_name: "log_user_step_example"
      arguments:
        user_action: "User requested a summary of AI news."
        timestamp: "2024-05-07T12:00:00Z"
      evidence_input_key: "user_action" # The 'user_action' input is the evidence.
  - plan: "Summarize the findings from E1 and the thought from E2."
    tool_call: null # Internal thought, using E1 and E2.
```

Query: {query}
Refinement History (if any):
{error_history}
"""
)

# Simplified prompt for refining a plan - YAML output
PLANNER_REFINE_PROMPT_TEMPLATE = PromptTemplate.from_template(
    """The previous plan attempt for the query failed validation or structure.
Original Query: {query}
Available Tools:
{tool_descriptions}

Previous Invalid Plan Output:
```yaml
{raw_plan_output}
```

Validation/Structure Errors:
{error_message}
{error_history}

**CRITICAL INSTRUCTIONS for Revising the Plan (Respond ONLY with a valid YAML plan block):**
1.  **Analyze Errors:** Carefully review the `Validation/Structure Errors` and `Previous Invalid Plan Output`.
2.  **Adhere to Tool Schema:** Ensure all `tool_call.arguments` strictly match the schema of the specified `tool_name` from `Available Tools`.
3.  **Correct YAML Structure:** Ensure the entire output is a single, valid YAML block starting with `steps:`.
4.  **Maintain Plan Intent:** Try to achieve the original plan's goal while fixing errors.
5.  **Use Exact Tool Names:** Tool names in `tool_call.tool_name` must be from `Available Tools`.
6.  **Specify Evidence Source (Based on Tool Purpose, Description, and Input Parameters):**
    *   Analyze the tool's `description`, its `Arguments Schema`, and **critically review any `Parameters explained` sections or detailed argument descriptions.** What is the primary *output* of the tool versus what is a significant *input that needs to be recorded as the primary outcome* of this step?
    *   **Standard Output Tools:** If the tool's description clearly indicates it **returns the primary valuable data**, then its direct output IS the evidence. **OMIT** `evidence_input_key` or set it to `null`.
    *   **Input as Evidence Tools (e.g., Logging, Thinking, Context-Setting Tools):**
        If the tool's primary purpose is to **process, log, record, or stage a specific input argument for later use or as the main result of this thinking/logging step**, and the tool itself might return only metadata, a status, or an ID, then the **value of that key input argument SHOULD be the evidence.**
        **CAREFULLY EXAMINE the `Arguments Schema` and `Parameters explained` section of the tool description.** 
        Look for parameters that represent the core content. For example, for a tool like `sequentialthinking` (with `thought` parameter), you **MUST set `evidence_input_key: "thought"`.**
        If such a parameter exists and it clearly holds the substantive information, **YOU MUST set `evidence_input_key` to that parameter's exact name.**
    *   **Decision Criteria:** If the tool's direct return value is merely status, metadata, or an ID, and a specific *input argument* holds the actual content/thought/data this step is designed to produce or record as its main outcome, **YOU MUST use `evidence_input_key` and set it to the name of that input argument.**
    *   If unclear, or if the tool's direct output genuinely seems to be the most valuable, omit `evidence_input_key`.

Example of Corrected YAML Structure:
```
```

If the error was `InvalidToolError` because `some_tool` was not in `Available Tools`, pick a valid tool.
If the error was `MissingEvidenceVariableError`, ensure every `tool_call` has an `evidence_variable`.
If the error was `IncorrectEvidenceSource` (hypothetical), ensure `evidence_input_key` is correctly used or omitted based on tool's purpose.

Respond ONLY with the corrected YAML plan block. No other text.
"""
)