from typing import Callable, Dict, Optional, Any

class ToolRegistry:
    """A simple registry for managing tools."""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._tool_metadata: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, name: str, tool_callable: Callable, description: Optional[str] = None, **kwargs):
        """
        Registers a tool (callable) with its name and optional metadata.

        Args:
            name: The unique name of the tool.
            tool_callable: The function or callable object implementing the tool.
            description: A brief description of what the tool does.
            **kwargs: Additional metadata (e.g., parameter schema).
        """
        if name in self._tools:
            print(f"Warning: Tool '{name}' already registered. Overwriting.")
        
        self._tools[name] = tool_callable
        self._tool_metadata[name] = {"description": description or "No description provided.", **kwargs}
        print(f"Tool '{name}' registered successfully.")

    def get_tool(self, name: str) -> Optional[Callable]:
        """
        Retrieves the callable for a registered tool by its name.

        Args:
            name: The name of the tool to retrieve.

        Returns:
            The tool callable if found, otherwise None.
        """
        return self._tools.get(name)

    def get_tool_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the metadata for a registered tool by its name.

        Args:
            name: The name of the tool.

        Returns:
            A dictionary containing the tool's metadata if found, otherwise None.
        """
        return self._tool_metadata.get(name)

    def list_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns a dictionary of all registered tools and their metadata.
        """
        return self._tool_metadata.copy()

    def get_all_tools(self) -> Dict[str, Callable]:
         """
         Returns a dictionary of all registered tool callables.
         """
         return self._tools.copy()

# Example usage (optional)
if __name__ == "__main__":
    registry = ToolRegistry()

    def sample_search(query: str):
        return f"Search results for: {query}"

    registry.register_tool("search", sample_search, description="Performs a web search.")
    registry.register_tool("calculator", lambda x, y: x + y, description="Adds two numbers.")

    print("\nAvailable Tools:")
    print(registry.list_tools())

    search_tool = registry.get_tool("search")
    if search_tool:
        print("\nExecuting search tool:")
        print(search_tool(query="LangChain"))

    calc_tool = registry.get_tool("calculator")
    if calc_tool:
         print("\nExecuting calculator tool:")
         print(calc_tool(5, 3))

    unknown_tool = registry.get_tool("unknown")
    print(f"\nGetting unknown tool: {unknown_tool}") 