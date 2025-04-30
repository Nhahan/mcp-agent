from langchain_core.prompts import PromptTemplate

# Prompt for the plan_parser node to correct malformed planner output
# Added placeholders for specific error context and tool schema if available
PLAN_CORRECTOR_PROMPT = PromptTemplate.from_template(
    """The following text is supposed to be a plan in YAML format, but it failed validation or parsing.
Correct the text so that it strictly adheres to the specified YAML structure AND the provided tool schema if applicable.

**Error Context (if available):**
{error_context}

**CRITICAL Instructions:**
1.  Fix the overall YAML structure (indentation, keys, lists, quotes, colons, null values).
2.  **Output `arguments` as a YAML mapping:** The value for the `arguments` key MUST be a correctly indented YAML mapping (which translates to a dictionary/object), NOT a string containing JSON. Do NOT enclose the braces `{{}}` in quotes.
3.  **Validate `arguments` content:** Ensure the keys and values within the `arguments` mapping are valid based on standard YAML/JSON rules (e.g., strings correctly quoted if needed).
4.  **Clean `arguments` content:** Remove any extraneous characters (like trailing '#', comments) found *within* the values of the mapping.
5.  **Apply Tool Schema (if provided):** If a specific tool schema is given below, ensure the keys and value types within the `arguments` YAML mapping for THAT tool call strictly match the schema (correct keys, types, required fields).
6.  Do NOT change the actual content (plan descriptions, tool names, values within `arguments`) unless necessary for structure, format, or schema adherence.
7.  Output ONLY the corrected YAML block, starting with `steps:` and nothing else before or after.

**Tool Schema for Correction (if applicable):**
Tool Name: {tool_name}
Schema (JSON):
```json
{tool_schema}
```

Expected YAML Structure (Illustrative):
```yaml
steps:
  - plan: "Description of the step (string)"
    tool_call: # Can be null or an object
      evidence_variable: "#E<number> (string)"
      tool_name: "tool_name_string"
      # IMPORTANT: arguments is a MAPPING, not a string literal
      arguments:
        key1: "string value" # Indented key-value pairs
        key2: 123           # Numbers don't need quotes
        key3: true          # Booleans don't need quotes
        # Ensure values match the tool's schema if provided
```

Malformed Input YAML:
```yaml
{raw_planner_output}
```

Corrected YAML Output:"""
)

# Add input_variables to reflect the new placeholders
PLAN_CORRECTOR_PROMPT.input_variables = ["raw_planner_output", "error_context", "tool_name", "tool_schema"]

if __name__ == "__main__":
    import json
    import yaml # Added for example parsing

    # Example with arguments as a bad string literal
    example_malformed = """
steps:
  - plan: Write a file with bad args string
    tool_call:
      evidence_variable: "#E1"
      tool_name: "write_file"
      arguments: '{{\"path\": \"/tmp/test.txt\", \"contnt\": \"hello world\"}}#' # String literal + bad key + trailing '#'
"""
    write_file_schema_example = {
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"]
    }
    error_context_example = "Pydantic validation failed: arguments field expects a dict, got str"

    formatted_prompt_with_schema = PLAN_CORRECTOR_PROMPT.format(
        raw_planner_output=example_malformed,
        error_context=error_context_example,
        tool_name="write_file",
        tool_schema=json.dumps(write_file_schema_example, indent=2)
    )
    print("--- Formatted Plan Corrector Prompt Example (Arguments as String) ---")
    print(formatted_prompt_with_schema)

    # Example expected correction: arguments is now a mapping
    expected_correction_mapping = """steps:
  - plan: "Write a file with bad args string"
    tool_call:
      evidence_variable: "#E1"
      tool_name: "write_file"
      arguments:
        path: "/tmp/test.txt"
        content: "hello world" """ # Corrected key, removed '#', value is YAML mapping
    print("\n--- Example Expected Corrected Output (Arguments as Mapping) ---")
    print(expected_correction_mapping)

    # Test parsing the expected correction
    try:
        parsed = yaml.safe_load(expected_correction_mapping)
        print("\n--- Parsing Test of Expected Output ---")
        print(f"Parsed type for arguments: {type(parsed['steps'][0]['tool_call']['arguments'])} ")
        print(json.dumps(parsed, indent=2))
        assert isinstance(parsed['steps'][0]['tool_call']['arguments'], dict)
        print("Parsing test successful!")
    except Exception as e:
        print(f"\nParsing test FAILED: {e}") 