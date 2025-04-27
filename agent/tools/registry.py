from typing import List, Dict, Optional, Callable, Coroutine, Any
from langchain_core.tools import Tool, BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field # Use Langchain's Pydantic for Tool args_schema
import json # For potential mock execution

from .loader import get_mcp_config, MCPConfig
from .models import MCPToolDefinition, MCPServerSpec

# Placeholder for actual tool execution logic (will be implemented later)
# This should ideally interact with the actual MCP server or a mock implementation.
async def execute_mcp_tool(server_name: str, tool_name: str, **kwargs) -> Any:
    """ Placeholder async function to simulate executing an MCP tool. """
    print(f"[Tool Execution] Simulating call to '{server_name}/{tool_name}' with args: {kwargs}")
    # In a real scenario, this would involve network calls to the MCP server.
    # For now, return a mock response based on the tool.
    if server_name == "web_search_placeholder" and tool_name == "search":
        query = kwargs.get("query", "no query provided")
        return f"Mock search results for: '{query}'"
    elif server_name == "another_server" and tool_name == "calculate":
        op1 = kwargs.get("operand1", 0)
        op2 = kwargs.get("operand2", 0)
        op = kwargs.get("operation", "add")
        if op == "add": return op1 + op2
        if op == "subtract": return op1 - op2
        # Add other operations as needed
        return "Unknown operation"
    return f"Mock response for {server_name}/{tool_name}"

# --- Pydantic Models for Tool Arguments ---
# Dynamically create Pydantic models for tool arguments based on MCP spec
# Store generated models to avoid re-creation
_tool_args_models: Dict[str, type[BaseModel]] = {}

def _create_args_model(tool_key: str, tool_spec: MCPToolDefinition) -> type[BaseModel]:
    """ Creates a Pydantic model for the arguments of a specific tool. """
    if tool_key in _tool_args_models:
        return _tool_args_models[tool_key]

    fields = {}
    for param_name, param_spec in tool_spec.parameters.items():
        # Basic type mapping (can be expanded)
        param_type = Any # Default
        if param_spec.type == "string":
            param_type = str
        elif param_spec.type == "number":
            param_type = float # Or int, depending on requirements
        elif param_spec.type == "boolean":
            param_type = bool
        # TODO: Handle array, object types if needed

        field_args = {"description": param_spec.description}
        if not param_spec.required:
            field_args["default"] = None
            param_type = Optional[param_type]

        fields[param_name] = (param_type, Field(**field_args))

    # Create the Pydantic model dynamically
    args_model = type(f"{tool_key.replace('/','_')}Args", (BaseModel,), fields)
    _tool_args_models[tool_key] = args_model
    return args_model

class ToolRegistry:
    """
    Manages the collection of available tools parsed from the MCP configuration.
    Converts MCP tool definitions into Langchain BaseTool objects.
    """
    def __init__(self, mcp_config: Optional[MCPConfig] = None):
        if mcp_config is None:
            mcp_config = get_mcp_config() # Load if not provided
        self.mcp_config = mcp_config
        self._tools: Optional[List[BaseTool]] = None
        self._tool_map: Optional[Dict[str, BaseTool]] = None

    def _create_langchain_tool(self, server_name: str, tool_name: str, tool_spec: MCPToolDefinition) -> BaseTool:
        """ Creates a Langchain Tool object from an MCP tool definition. """
        tool_key = f"{server_name}/{tool_name}"
        args_schema = _create_args_model(tool_key, tool_spec)

        # Define the async execution function (coroutine)
        async def _acoroutine(**kwargs):
            # Here we'd call the actual MCP server execution logic
            # For now, we call the placeholder
            return await execute_mcp_tool(server_name, tool_name, **kwargs)

        # Create the Langchain Tool
        # Note: We provide both func and coroutine, Langchain prefers coroutine if available
        return Tool(
            name=tool_key, # Unique name combining server and tool
            description=tool_spec.description,
            func=None, # No synchronous version provided for now
            coroutine=_acoroutine, # Async execution function
            args_schema=args_schema # Pydantic model for arguments
        )

    def load_tools(self) -> List[BaseTool]:
        """ Parses the MCP config and creates a list of Langchain tools. Caches the result. """
        if self._tools is None:
            self._tools = []
            self._tool_map = {}
            for server_name, server_spec in self.mcp_config.mcpServers.items():
                for tool_name, tool_spec in server_spec.tools.items():
                    langchain_tool = self._create_langchain_tool(server_name, tool_name, tool_spec)
                    self._tools.append(langchain_tool)
                    self._tool_map[langchain_tool.name] = langchain_tool
            print(f"Loaded {len(self._tools)} tools into registry.")
        return self._tools

    def get_tools(self) -> List[BaseTool]:
        """ Returns the list of loaded Langchain tools. Loads them if not already loaded. """
        return self.load_tools()

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """ Returns a specific tool by its name (e.g., 'server_name/tool_name'). """
        if self._tool_map is None:
            self.load_tools()
        return self._tool_map.get(name)

# Example Usage
if __name__ == "__main__":
    import asyncio

    async def run_example():
        try:
            registry = ToolRegistry()
            tools = registry.get_tools()

            print("\nAvailable Tools:")
            for tool in tools:
                print(f"- {tool.name}: {tool.description}")
                if tool.args_schema:
                    print(f"  Args Schema: {tool.args_schema.schema()}") # Print Pydantic schema

            # Get a specific tool
            search_tool = registry.get_tool("web_search_placeholder/search")
            if search_tool:
                print(f"\nGot tool: {search_tool.name}")
                # Example invocation (using the placeholder coroutine)
                print("Testing tool invocation...")
                result = await search_tool.arun({"query": "Langchain ReWOO"}) # Use arun for async Tool
                print(f"Invocation Result: {result}")

                # Example with missing required arg (should ideally be caught by Langchain internals or the execution logic)
                # result_no_arg = await search_tool.arun({})
                # print(f"Invocation Result (no arg): {result_no_arg}")

                # Example of calculate tool
                calc_tool = registry.get_tool("another_server/calculate")
                if calc_tool:
                     result_calc = await calc_tool.arun({"operand1": 5, "operand2": 3, "operation": "subtract"})
                     print(f"Calc Result: {result_calc}")

        except Exception as e:
            print(f"\nAn error occurred during registry example: {e}")

    asyncio.run(run_example()) 