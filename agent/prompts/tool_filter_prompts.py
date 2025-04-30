from langchain_core.prompts import PromptTemplate

TOOL_FILTER_PROMPT_TEMPLATE = PromptTemplate.from_template(
    """Given the user query and the list of available tools with their simplified descriptions, identify the tools **most likely necessary** to fulfill the query accurately and completely.

Available Tools (Name: Simplified Description):
{tool_descriptions}

User Query: {query}

Instructions:
1. Analyze the user query carefully to understand the **primary goal** and any **explicit secondary tasks**.
2. Review the available tools and their simplified descriptions.
3. Select ONLY the tools that **directly contribute** to achieving the primary goal OR fulfill a secondary task **explicitly mentioned or strongly implied** in the query. 
4. **Be selective:** Avoid including tools that might seem broadly related but are not strictly required to complete the specific tasks identified in the query. Focus on the tools essential for *this* particular request.
5. Output MUST be ONLY a JSON list containing the exact string names of the selected tools. 
   - Example: ["tool1_name", "tool2_name", "another_tool"]
   - DO NOT include any text before or after the JSON list.
   - DO NOT make up tool names. Use only names from the 'Available Tools' list.
6. If no tools from the list are clearly necessary for the query, output an empty JSON list: [].

Necessary Tool Names (JSON List Only):"""
)

if __name__ == "__main__":
    # Example for testing the prompt
    example_query = "Write 'hello world' to file.txt and then read it back."
    example_tools_desc = """
- file/write: Writes content to a file.
- file/read: Reads content from a file.
- web/search: Searches the web.
- internal/reasoning: Performs internal thought process.
"""
    
    formatted = TOOL_FILTER_PROMPT_TEMPLATE.format(
        tool_descriptions=example_tools_desc,
        query=example_query
    )
    print("--- Formatted Tool Filter Prompt --- ")
    print(formatted)
    print("\n--- Expected Output --- ")
    print('["file/write", "file/read"]') 