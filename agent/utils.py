import logging
import json
from typing import List, Dict, Optional
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

# --- Tool Description Formatting --- #
def format_tool_descriptions_with_schema(tools: List[BaseTool]) -> str:
    """Formats tool descriptions including their argument schemas."""
    if not tools:
        return "No tools available."
    
    formatted_descriptions = []
    for tool in tools:
        # Use the tool name directly as provided by the client
        tool_display_name = tool.name 
        description = f"- {tool_display_name}: {tool.description}"
        schema_str = "No arguments required."
        schema_dict: Optional[Dict] = None

        # 1. Check if args_schema exists and is a dictionary
        if hasattr(tool, 'args_schema') and isinstance(tool.args_schema, dict):
            schema_dict = tool.args_schema
            logger.debug(f"Found schema for {tool_display_name} in args_schema (type: dict)")
        # 2. Check if args exists (as fallback or alternative)
        elif hasattr(tool, 'args') and isinstance(tool.args, dict):
            schema_dict = tool.args # Use args if args_schema is not a dict but args is
            logger.debug(f"Found schema for {tool_display_name} in args (type: dict)")
        # 3. Check if args_schema exists but is NOT a dict (maybe Pydantic? Unlikely based on error)
        elif hasattr(tool, 'args_schema') and tool.args_schema:
             try:
                 # Attempt to call .schema() assuming it might be Pydantic
                 schema_dict_from_method = tool.args_schema.schema()
                 if isinstance(schema_dict_from_method, dict):
                     schema_dict = schema_dict_from_method
                     logger.debug(f"Found schema for {tool_display_name} via args_schema.schema() call")
                 else:
                      logger.warning(f"args_schema.schema() for {tool_display_name} did not return a dict.")
             except AttributeError:
                 logger.warning(f"args_schema found for {tool_display_name} but it's not a dict and has no .schema() method. Type: {type(tool.args_schema)}")
             except Exception as e:
                 logger.warning(f"Error calling .schema() on args_schema for {tool_display_name}: {e}")
        
        # Process the found schema dictionary
        if schema_dict:
            try:
                # Simplify schema slightly for prompt - focus on properties and required
                simplified_schema = {
                    "properties": schema_dict.get("properties", {}),
                    "required": schema_dict.get("required", [])
                }
                # Only add schema string if there are properties
                if simplified_schema["properties"]:
                     schema_str = f"Arguments Schema (JSON): {json.dumps(simplified_schema)}"
                # else schema_str remains "No arguments required."
            except Exception as e:
                 logger.error(f"Error processing schema dictionary for tool {tool_display_name}: {e}. Schema was: {schema_dict}", exc_info=True)
                 schema_str = f"Arguments Schema (Error processing: {e})"

        formatted_descriptions.append(f"{description}\n  {schema_str}")
        
    return "\n".join(formatted_descriptions) 

# --- Simplified Tool Description Formatting --- #
def format_tool_descriptions_simplified(tools: List[BaseTool]) -> str:
    """Formats tool descriptions with only name and the first sentence."""
    if not tools:
        return "No tools available."
    
    formatted_descriptions = []
    for tool in tools:
        # Use the tool name directly
        tool_display_name = tool.name 
        
        first_sentence = tool.description.split('.')[0].split('\n')[0].strip()
        if not first_sentence:
            first_sentence = tool.description.split('\n')[0].strip()
        if len(first_sentence) > 100:
             first_sentence = first_sentence[:100] + "..."
             
        description = f"- {tool_display_name}: {first_sentence}"
        formatted_descriptions.append(description)
        
    return "\n".join(formatted_descriptions) 