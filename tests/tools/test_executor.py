# tests/tools/test_executor.py
import pytest
from agent.tools.executor import get_tool_executor, MCPToolExecutor, BaseToolExecutor

# Mark all tests in this module as async
pytestmark = pytest.mark.asyncio

async def test_get_tool_executor_returns_instance():
    """ Test that get_tool_executor returns a valid executor instance. """
    executor = get_tool_executor()
    assert executor is not None
    assert isinstance(executor, BaseToolExecutor)
    # Check if it's the specific placeholder implementation for now
    assert isinstance(executor, MCPToolExecutor)

async def test_executor_execute_mock_search():
    """ Test executing the mock web search tool. """
    executor = get_tool_executor()
    query = "testing executor"
    expected_result = f"Mock search results for: '{query}'"
    result = await executor.execute("web_search_placeholder", "search", {"query": query})
    assert result == expected_result

async def test_executor_execute_mock_calculate():
    """ Test executing the mock calculate tool. """
    executor = get_tool_executor()
    params = {"operand1": 15, "operand2": 8, "operation": "subtract"}
    expected_result = 7
    result = await executor.execute("another_server", "calculate", params)
    assert result == expected_result

    params_add = {"operand1": 10, "operand2": 5} # Default operation is add
    expected_result_add = 15
    result_add = await executor.execute("another_server", "calculate", params_add)
    assert result_add == expected_result_add

async def test_executor_execute_not_implemented():
    """ Test executing a tool not defined in the mock implementation raises NotImplementedError. """
    executor = get_tool_executor()
    with pytest.raises(NotImplementedError, match="execution not implemented in this placeholder"):
        await executor.execute("non_existent_server", "some_tool", {})

    # Also test a known server but unknown tool
    with pytest.raises(NotImplementedError, match="execution not implemented in this placeholder"):
        await executor.execute("web_search_placeholder", "unknown_action", {}) 