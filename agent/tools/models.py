from typing import Dict, Optional, Any
from pydantic import BaseModel, Field

class MCPToolParameter(BaseModel):
    """ Represents a single parameter for an MCP tool. """
    description: str
    type: str # Consider using Literal['string', 'number', 'boolean', 'array', 'object'] for stricter typing
    required: Optional[bool] = False

class MCPToolReturns(BaseModel):
    """ Represents the return value specification for an MCP tool. """
    description: str
    type: str # Consider using Literal or more complex types

class MCPToolDefinition(BaseModel):
    """ Defines a single tool within an MCP server specification. """
    description: str
    parameters: Dict[str, MCPToolParameter] = Field(default_factory=dict)
    returns: MCPToolReturns # Assuming a tool always returns something

class MCPServerSpec(BaseModel):
    """ Represents the specification for a single MCP server. """
    description: str
    tools: Dict[str, MCPToolDefinition]

class MCPConfig(BaseModel):
    """ Represents the overall structure of the mcp.json file. """
    mcpServers: Dict[str, MCPServerSpec]

# Example Usage (for testing or understanding)
if __name__ == "__main__":
    # Example mcp.json data structure
    example_data = {
        "mcpServers": {
            "web_search_placeholder": {
                "description": "A placeholder tool that simulates web search.",
                "tools": {
                    "search": {
                        "description": "Performs a mock web search for the given query.",
                        "parameters": {
                            "query": {
                                "description": "The search query.",
                                "type": "string",
                                "required": True
                            }
                        },
                        "returns": {
                            "description": "Mock search results.",
                            "type": "string"
                        }
                    }
                }
            },
            "another_server": {
                "description": "Another example server.",
                "tools": {
                    "calculate": {
                        "description": "Performs a calculation.",
                        "parameters": {
                            "operand1": {"description": "First number", "type": "number", "required": True},
                            "operand2": {"description": "Second number", "type": "number", "required": True},
                            "operation": {"description": "Operation type", "type": "string", "required": False}
                        },
                        "returns": {
                            "description": "Calculation result.",
                            "type": "number"
                        }
                    }
                }
            }
        }
    }

    # Parse the example data using the Pydantic model
    try:
        mcp_config = MCPConfig.model_validate(example_data)
        print("Successfully parsed example MCP config:")
        print(mcp_config.model_dump_json(indent=2))

        # Accessing data
        search_tool = mcp_config.mcpServers["web_search_placeholder"].tools["search"]
        print(f"\nSearch tool description: {search_tool.description}")
        print(f"Search query parameter type: {search_tool.parameters['query'].type}")
        print(f"Calculate tool return type: {mcp_config.mcpServers['another_server'].tools['calculate'].returns.type}")

    except Exception as e:
        print(f"Error parsing example data: {e}") 