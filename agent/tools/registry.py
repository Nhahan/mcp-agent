from typing import List, Dict, Optional, Coroutine, Any
from langchain_core.tools import Tool, BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field
import json

from .loader import get_mcp_config, MCPConfig
from .models import MCPToolDefinition, MCPServerSpec
# Import the executor interface and getter
from .executor import get_tool_executor, BaseToolExecutor

# # Placeholder for actual tool execution logic (REMOVED - Now uses executor)
# async def execute_mcp_tool(server_name: str, tool_name: str, **kwargs) -> Any:
#     ...

# --- Pydantic Models for Tool Arguments ---
_tool_args_models: Dict[str, type[BaseModel]] = {}

def _create_args_model(tool_key: str, tool_spec: MCPToolDefinition) -> type[BaseModel]:
    """ Creates a Pydantic model for the arguments of a specific tool. """
    if tool_key in _tool_args_models:
        return _tool_args_models[tool_key]

    fields = {}
    for param_name, param_spec in tool_spec.parameters.items():
        param_type = Any
        if param_spec.type == "string": param_type = str
        elif param_spec.type == "number": param_type = float
        elif param_spec.type == "boolean": param_type = bool

        field_args = {"description": param_spec.description}
        if not param_spec.required:
            field_args["default"] = None
            param_type = Optional[param_type]

        fields[param_name] = (param_type, Field(**field_args))

    args_model = type(f"{tool_key.replace('/','_')}Args", (BaseModel,), fields)
    _tool_args_models[tool_key] = args_model
    return args_model

class ToolRegistry:
    """
    Manages the collection of available tools parsed from the MCP configuration.
    Converts MCP tool definitions into Langchain BaseTool objects.
    """
    def __init__(self, mcp_config: Optional[MCPConfig] = None, executor: Optional[BaseToolExecutor] = None):
        if mcp_config is None:
            mcp_config = get_mcp_config()
        self.mcp_config = mcp_config
        # Get the executor instance (singleton or injected)
        self.executor = executor if executor is not None else get_tool_executor()
        self._tools: Optional[List[BaseTool]] = None
        self._tool_map: Optional[Dict[str, BaseTool]] = None

    def _create_langchain_tool(self, server_name: str, tool_name: str, tool_spec: MCPToolDefinition) -> BaseTool:
        """ Creates a Langchain Tool object from an MCP tool definition. """
        tool_key = f"{server_name}/{tool_name}"
        args_schema = _create_args_model(tool_key, tool_spec)

        # Define the async execution function (coroutine) using the executor
        async def _acoroutine(**kwargs):
            # Use the injected executor instance
            return await self.executor.execute(server_name, tool_name, kwargs)

        return Tool(
            name=tool_key,
            description=tool_spec.description,
            func=None,
            coroutine=_acoroutine,
            args_schema=args_schema
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

# Example Usage (remains mostly the same, now uses the executor implicitly)
if __name__ == "__main__":
    import asyncio

    async def run_example():
        try:
            # Registry now implicitly uses the executor from get_tool_executor()
            registry = ToolRegistry()
            tools = registry.get_tools()

            print("\nAvailable Tools:")
            for tool in tools:
                print(f"- {tool.name}: {tool.description}")
                if tool.args_schema:
                    print(f"  Args Schema: {tool.args_schema.schema()}")

            search_tool = registry.get_tool("web_search_placeholder/search")
            if search_tool:
                print(f"\nGot tool: {search_tool.name}")
                print("Testing tool invocation (via registry/executor)...")
                result = await search_tool.arun({"query": "Langchain ReWOO via Registry"})
                print(f"Invocation Result: {result}")

            calc_tool = registry.get_tool("another_server/calculate")
            if calc_tool:
                print("\nTesting calculate tool (via registry/executor)...")
                result_calc = await calc_tool.arun({"operand1": 25, "operand2": 5, "operation": "add"})
                print(f"Calc Result: {result_calc}")

        except Exception as e:
            print(f"\nAn error occurred during registry example: {e}")

    asyncio.run(run_example()) 