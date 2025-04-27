# agent/graph.py
from typing import List, Dict, Any, Optional, AsyncGenerator
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from functools import partial

# Import state definition and component functions
from .state import ReWOOState, PlanStep, Evidence, ToolResult, ToolCall
from .planner import generate_plan
from .solver import generate_final_answer
# Import MCP client from adapter library
from langchain_mcp_adapters.client import MultiServerMCPClient

# Helper function to find a tool by name from the list
def find_tool_by_name(tools: List[BaseTool], name: str) -> Optional[BaseTool]:
    for tool in tools:
        if tool.name == name:
            return tool
    return None

# --- LangGraph Nodes ---

async def planner_node(state: ReWOOState, llm: BaseLanguageModel, tools: List[BaseTool]) -> Dict[str, Any]:
    """ Generates the plan based on the input query. """
    print("\n--- Running Planner Node ---")
    query = state['input_query']
    try:
        plan_steps = await generate_plan(llm, tools, query)
        # Initialize state fields needed for the graph execution
        return {
            "plan": plan_steps,
            "current_step_index": 0,
            "evidence": [],
            "final_answer": None,
            "error": None
        }
    except Exception as e:
        print(f"Error in Planner Node: {e}")
        return {"error": f"Planner failed: {e}", "plan": [], "evidence": []} # Ensure keys exist even on error

async def worker_node(state: ReWOOState, tools: List[BaseTool]) -> Dict[str, Any]:
    """ Executes the tool call for the current plan step or advances index. """
    current_step_index = state.get('current_step_index', 0)
    plan = state.get('plan', [])
    evidence = state.get('evidence', [])
    print(f"\n--- Running Worker Node (Step {current_step_index + 1}/{len(plan)}) ---")


    if current_step_index >= len(plan):
        print("Worker Node: Plan already finished according to index.")
        # This path shouldn't be reached if routing logic is correct, but handle defensively
        return {}

    current_step: PlanStep = plan[current_step_index]
    tool_call: Optional[ToolCall] = current_step.get('tool_call')

    if not tool_call:
        print(f"Worker Node: No tool call required for step {current_step_index + 1}. Advancing index.")
        # Just advance the index if no tool is needed for this step
        return {"current_step_index": current_step_index + 1}

    tool_name = tool_call['tool_name']
    tool_args = tool_call['arguments']

    # Substitute evidence variables (#E<n>) in arguments
    try:
        print(f"Worker Node: Attempting to substitute evidence in args: {tool_args}")
        substituted_args = _substitute_evidence(tool_args, evidence)
        print(f"Worker Node: Substituted args: {substituted_args}")
    except Exception as e:
        print(f"Error substituting evidence for step {current_step_index + 1}: {e}")
        # Record the error as evidence for this step
        error_result = ToolResult(tool_name=tool_name, output=None, error=f"Evidence substitution failed: {e}")
        new_evidence = Evidence(step_index=current_step_index + 1, tool_result=error_result, processed_evidence=f"Error: {e}")
        return {"evidence": evidence + [new_evidence], "current_step_index": current_step_index + 1}


    print(f"Worker Node: Executing tool '{tool_name}' with args: {substituted_args}")
    target_tool = find_tool_by_name(tools, tool_name)

    if not target_tool:
        print(f"Worker Node: Tool '{tool_name}' not found.")
        error_result = ToolResult(tool_name=tool_name, output=None, error=f"Tool '{tool_name}' not found.")
        new_evidence = Evidence(step_index=current_step_index + 1, tool_result=error_result, processed_evidence=f"Error: Tool not found")
    else:
        try:
            # Execute the tool (Langchain Tool's coroutine handles execution, adapter handles MCP call)
            tool_output = await target_tool.arun(substituted_args)
            print(f"Worker Node: Tool '{tool_name}' output: {tool_output}")
            tool_result = ToolResult(tool_name=tool_name, output=tool_output, error=None)
            # TODO: Implement 'processed_evidence' generation (potentially another LLM call)
            # For now, just use the raw output or a summary
            processed_evidence_str = f"Executed {tool_name}, output: {str(tool_output)[:200]}..." # Simple summary
            new_evidence = Evidence(step_index=current_step_index + 1, tool_result=tool_result, processed_evidence=processed_evidence_str)
        except Exception as e:
            import traceback
            print(f"Worker Node: Error executing tool '{tool_name}': {e}")
            traceback.print_exc() # Print stack trace for debugging
            error_result = ToolResult(tool_name=tool_name, output=None, error=str(e))
            new_evidence = Evidence(step_index=current_step_index + 1, tool_result=error_result, processed_evidence=f"Error executing tool: {e}")

    # Add the new evidence and move to the next step index
    return {"evidence": evidence + [new_evidence], "current_step_index": current_step_index + 1}


def _substitute_evidence(args: Dict[str, Any], evidence_list: List[Evidence]) -> Dict[str, Any]:
    """ Substitutes #E<n> placeholders in args with actual evidence output. Handles nested structures. """
    # Create a map from placeholder (e.g., #E1) to the corresponding tool output
    evidence_map = {f"#E{e['step_index']}": e['tool_result']['output'] 
                    for e in evidence_list if not e['tool_result'].get('error')}
    print(f"Evidence map for substitution: {evidence_map}")

    def recursive_substitute(value: Any) -> Any:
        if isinstance(value, str):
            if value.startswith("#E"):
                if value in evidence_map:
                    print(f"Substituting placeholder '{value}'")
                    return evidence_map[value]
                else:
                    # Allow missing evidence for flexibility, return placeholder
                    print(f"Warning: Evidence placeholder '{value}' not found in collected evidence: {list(evidence_map.keys())}. Keeping placeholder.")
                    # raise ValueError(f"Evidence placeholder '{value}' not found in collected evidence: {list(evidence_map.keys())}")
                    return value
            else:
                return value
        elif isinstance(value, dict):
            return {k: recursive_substitute(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [recursive_substitute(item) for item in value]
        else:
            return value # Handle numbers, booleans, etc.

    return recursive_substitute(args)


async def solver_node(state: ReWOOState, llm: BaseLanguageModel) -> Dict[str, Any]:
    """ Generates the final answer based on the collected evidence. """
    print("\n--- Running Solver Node ---")
    if state.get("error"): # If there was an error earlier, skip solving
         print("Solver Node: Skipping due to previous error.")
         # Even if skipping, ensure final_answer key exists if graph expects it
         return {"final_answer": state.get("final_answer")}
    try:
        final_answer = await generate_final_answer(llm, state)
        return {"final_answer": final_answer}
    except Exception as e:
        print(f"Error in Solver Node: {e}")
        # Return error state, but ensure keys graph expects are present
        return {"error": f"Solver failed: {e}", "final_answer": state.get("final_answer")}

# --- Conditional Edge Logic ---

def should_continue(state: ReWOOState) -> str:
    """ Determines whether to continue the plan execution or move to the solver. """
    print("\n--- Checking Condition: Should Continue? ---")
    if state.get("error"): # Check if any node set an error
         print("Condition: Error detected in state, routing to END.")
         return "__end__" # Use LangGraph's END sentinel

    plan = state.get('plan', [])
    current_step_index = state.get('current_step_index', 0)

    if not plan: # Handle case where planner failed to produce a plan
        print("Condition: No plan found, routing to END.")
        return "__end__"

    if current_step_index >= len(plan):
        print(f"Condition: Plan finished (Index {current_step_index} >= Plan length {len(plan)}), routing to Solver.")
        return "solve"
    else:
        # Always route to worker if plan is not finished
        print(f"Condition: Plan not finished (Index {current_step_index} < Plan length {len(plan)}), routing to Worker.")
        return "work"


# --- Graph Construction ---

def create_rewoo_graph(llm: BaseLanguageModel, tools: List[BaseTool]) -> StateGraph:
    """ Creates and configures the LangGraph StateGraph for the ReWOO agent. """
    builder = StateGraph(ReWOOState)

    # Bind the tools list to the worker node using partial
    # Planner also needs tools to inform the plan generation
    bound_planner_node = partial(planner_node, llm=llm, tools=tools)
    bound_worker_node = partial(worker_node, tools=tools) # Pass tools here
    bound_solver_node = partial(solver_node, llm=llm)

    builder.add_node("planner", bound_planner_node)
    builder.add_node("worker", bound_worker_node)
    builder.add_node("solver", bound_solver_node)

    # Define edges
    builder.set_entry_point("planner")

    # After planner, decide where to go (usually worker or end if planner failed)
    builder.add_conditional_edges(
        "planner",
        lambda state: "__end__" if state.get("error") else "worker",
        {"worker": "worker", "__end__": END}
    )

    # Conditional edge after worker
    builder.add_conditional_edges(
        "worker",
        should_continue, # Function determines the next step
        {
            "work": "worker",   # Loop back to worker for the next step
            "solve": "solver",  # Finish plan, go to solver
            "__end__": END    # Go to end if error occurred or condition dictates
        }
    )

    builder.add_edge("solver", END) # Go to end after solving

    # Compile the graph
    graph = builder.compile()
    print("ReWOO graph compiled successfully.")
    return graph

# --- MCP Client and Tool Loading Context Manager ---

# Define your MCP server configurations here
# IMPORTANT: Replace placeholders like '/path/to/' with actual absolute paths
#            or use environment variables / configuration files.
# Example using taskmaster-ai via npx (adjust based on your setup)
# NOTE: The 'command' and 'args' will depend heavily on how taskmaster-ai
#       is installed and intended to be run as an MCP server.
#       This example assumes it can be run directly via npx.
#       If it needs specific flags to run in MCP server mode, add them to 'args'.
MCP_SERVER_CONFIG = {
    "taskmaster": {
        "command": "npx", # Or 'node', 'python', etc. depending on taskmaster
        "args": ["task-master-ai", "--mcp-stdio"], # Assuming a flag like --mcp-stdio exists
         # If taskmaster needs to run from a specific directory:
        # "cwd": "/Users/sunningkim/Developer/mcp-agent",
        "transport": "stdio",
    },
    # Add other MCP servers here if needed
    # "another_server": { ... }
}

@asynccontextmanager
async def make_graph_with_mcp_tools(llm: BaseLanguageModel) -> AsyncGenerator[StateGraph, None]:
    """ Context manager to handle MCP client lifecycle and graph creation. """
    print("Initializing MultiServerMCPClient...")
    async with MultiServerMCPClient(MCP_SERVER_CONFIG) as client:
        print("MCP Client Initialized. Loading tools...")
        try:
            # Fetch tools from the active client session
            tools = await client.get_tools()
            print(f"Loaded {len(tools)} tools from MCP servers:")
            for tool in tools:
                print(f" - {tool.name}: {tool.description}") # Tool name includes server prefix

            if not tools:
                print("Warning: No tools loaded from MCP servers. Agent may lack capabilities.")

            # Create the graph instance using the loaded tools
            graph = create_rewoo_graph(llm, tools)
            yield graph # Yield the compiled graph to be used

        except Exception as e:
            print(f"Error loading MCP tools or creating graph: {e}")
            import traceback
            traceback.print_exc()
            # Handle error appropriately, maybe raise or yield None/error state
            raise # Re-raise the exception to signal failure

        finally:
            print("MCP Client context exited.") # Cleanup is handled by MultiServerMCPClient context


# --- Get Compiled Graph ---
_compiled_graph = None
_graph_context_active = False # Flag to prevent re-entry if used incorrectly

async def get_agent_executor(llm: Optional[BaseLanguageModel] = None, force_reload: bool = False):
    """
    Loads dependencies (LLM), manages MCP tools, and returns the compiled LangGraph executor.

    Args:
        llm: Optional language model instance. If None, loads using core.llm_loader.
        force_reload: If True, forces recompiling the graph (MCP client restarts on each call now).

    Returns:
        A compiled LangGraph runnable instance.
    """
    # Note: Caching the graph instance (_compiled_graph) is tricky now because
    # the MCP client and tools are live within the context manager.
    # Each call to get_agent_executor will likely need to recreate the client and graph.
    # Consider managing the client lifecycle at a higher level if performance is critical.
    global _compiled_graph, _graph_context_active
    # Simple re-entry check
    if _graph_context_active:
        raise RuntimeError("Graph context is already active. Ensure proper async handling.")


    print("Initializing agent executor...")
    # 1. Load LLM
    if llm is None:
        try:
             from core.llm_loader import load_llm as core_load_llm
             llm = core_load_llm()
        except ImportError:
            print("Error: Could not import core.llm_loader. Make sure it exists and path is correct.")
            raise
        except Exception as e:
            print(f"Fatal Error: Could not load LLM. {e}")
            raise

    # 2. Load Tools and Create Graph using Context Manager
    try:
        _graph_context_active = True
        # Use the async context manager to get the graph with live tools
        # This part needs careful handling in the application's main async loop
        # We return the context manager itself or handle it here.
        # For simplicity in this function, we'll enter the context and return the graph,
        # but the caller needs to be aware the context needs managing.
        # A better pattern might be to return the *async generator* from make_graph_with_mcp_tools
        # and let the caller manage the `async for graph in ...:` loop.

        # Let's return the graph directly after creating it within the context
        # This implies the client lives only during this call - likely incorrect for agent execution.
        # ----
        # Revised approach: get_agent_executor should return the ready-to-run graph *instance*.
        # The context management needs to happen *around* the agent's invocation.
        # How to achieve this?
        # Option A: Pass the client around. Complicates node signatures.
        # Option B: Use the context manager (`make_graph_with_mcp_tools`) higher up in the application entry point.

        # Let's assume Option B is implemented by the caller.
        # This function will now just return the *context manager function*.
        # The caller will do: `async with make_graph_with_mcp_tools(llm) as agent_executor:`
        # return make_graph_with_mcp_tools(llm) # Return the async context manager function

        # --- OR --- Modify to return the compiled graph directly for now, assuming caller manages context
        # This requires running the context manager here temporarily.
        async with make_graph_with_mcp_tools(llm) as graph:
             _compiled_graph = graph # Store the graph instance
             print("Agent executor initialized with MCP tools.")
             _graph_context_active = False
             return _compiled_graph # Return the graph compiled with live tools

    except Exception as e:
        _graph_context_active = False
        print(f"Fatal Error: Could not initialize agent executor with MCP tools. {e}")
        raise


# Example Usage - Needs significant changes due to async context management
if __name__ == "__main__":
    # import asyncio # Already imported
    from langchain_community.llms.fake import FakeListLLM

    async def run_graph_example():
        print("--- Running Graph Example with MCP Adapters (Conceptual) ---")
        # --- Setup Fake LLM ---
        # Note: Mocking MCP tool calls is complex as they happen via the client.
        # For a real test, you'd need mock MCP servers or skip tool execution testing here.
        # Define the fake plan and final answer carefully, ensuring valid syntax
        fake_plan_response = (
            'Plan: Use taskmaster to list tasks.\\n'
            '#E1 = taskmaster/mcp_taskmaster-ai_get_tasks[{\"projectRoot\": \"/Users/sunningkim/Developer/mcp-agent\"}]'
        )
        fake_solver_response = 'Final Answer: Got tasks from taskmaster (Evidence 1).'

        fake_llm = FakeListLLM(responses=[
            fake_plan_response,
            fake_solver_response
        ])

        try:
            print("Getting agent executor context manager...")
            # The executor is now obtained via the context manager
            async with make_graph_with_mcp_tools(fake_llm) as agent_executor:
                print("Agent executor obtained within context.")
                if agent_executor is None:
                     raise RuntimeError("Failed to create agent executor.")

                print("\nInvoking agent graph...")
                initial_state = {"input_query": "List tasks using taskmaster"}
                # Initialize state keys
                initial_state.setdefault('plan', [])
                initial_state.setdefault('current_step_index', 0)
                initial_state.setdefault('evidence', [])
                initial_state.setdefault('final_answer', None)
                initial_state.setdefault('error', None)


                # IMPORTANT: This example assumes the 'taskmaster' MCP server defined
                # in MCP_SERVER_CONFIG is running and accessible via npx.
                # If not, the tool call within the graph will fail.
                final_state = await agent_executor.ainvoke(initial_state)

                print("\n--- Final State --- ")
                import json
                print(json.dumps({
                    "input_query": final_state.get('input_query'),
                    "plan_steps": len(final_state.get('plan', [])),
                    "evidence_collected": len(final_state.get('evidence', [])),
                    "final_answer": final_state.get('final_answer'),
                    "error": final_state.get('error')
                }, indent=2))

                # Add assertions based on expected flow (may fail if MCP server isn't running)
                # assert final_state.get('final_answer') is not None
                # assert final_state.get('error') is None

        except Exception as e:
            print(f"\nAn error occurred during graph example: {e}")
            import traceback
            traceback.print_exc()

    # Run the async example
    asyncio.run(run_graph_example())

    # Removed old example using ToolRegistry 