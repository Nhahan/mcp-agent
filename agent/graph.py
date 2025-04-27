# agent/graph.py
from typing import List, Dict, Any, Optional
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END

# Import state definition and component functions
from .state import ReWOOState, PlanStep, Evidence, ToolResult, ToolCall
from .planner import generate_plan
from .solver import generate_final_answer
# Import tool execution logic (using the registry and executor implicitly via Tool objects)
from .tools.registry import ToolRegistry

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
            # Execute the tool (Langchain Tool's coroutine handles execution)
            # Use apply_and_parse or similar if tool needs specific output parsing
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
                    raise ValueError(f"Evidence placeholder '{value}' not found in collected evidence: {list(evidence_map.keys())}")
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

    # Add nodes, binding the llm and tools where needed using partial or lambda
    # Ensure node functions return dictionaries with keys corresponding to ReWOOState fields
    builder.add_node("planner", lambda s: planner_node(s, llm, tools))
    builder.add_node("worker", lambda s: worker_node(s, tools))
    builder.add_node("solver", lambda s: solver_node(s, llm))

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

# --- Get Compiled Graph ---
_compiled_graph = None

def get_agent_executor(llm: Optional[BaseLanguageModel] = None, force_reload: bool = False):
    """
    Loads dependencies (LLM, Tools) and returns the compiled LangGraph executor.

    Args:
        llm: Optional language model instance. If None, loads using core.llm_loader.
        force_reload: If True, forces reloading of tools and recompiling the graph.

    Returns:
        A compiled LangGraph runnable instance.
    """
    global _compiled_graph
    if _compiled_graph is not None and not force_reload:
        print("Returning cached agent executor.")
        return _compiled_graph

    print("Initializing agent executor...")
    # 1. Load LLM
    if llm is None:
        # Ensure core.llm_loader is imported only when needed
        try:
             from core.llm_loader import load_llm as core_load_llm
             llm = core_load_llm()
        except ImportError:
            print("Error: Could not import core.llm_loader. Make sure it exists and path is correct.")
            raise
        except Exception as e:
            print(f"Fatal Error: Could not load LLM. {e}")
            raise # Re-raise critical error

    # 2. Load Tools
    try:
        registry = ToolRegistry()
        tools = registry.get_tools()
        if not tools:
             print("Warning: No tools loaded from ToolRegistry.")
    except ImportError:
        print("Error: Could not import ToolRegistry. Make sure agent.tools components exist.")
        raise
    except Exception as e:
        print(f"Fatal Error: Could not load Tools. {e}")
        raise # Re-raise critical error

    # 3. Create and Compile Graph
    try:
        _compiled_graph = create_rewoo_graph(llm, tools)
    except Exception as e:
        print(f"Fatal Error: Could not compile ReWOO graph. {e}")
        raise # Re-raise critical error

    print("Agent executor initialized.")
    return _compiled_graph


# Example Usage
if __name__ == "__main__":
    import asyncio
    from langchain_community.llms.fake import FakeListLLM

    async def run_graph_example():
        # --- Setup Fake LLM and Mocks ---
        fake_plan_output = """
Plan: Need to find out what LangGraph is.
#E1 = web_search_placeholder/search[{"query": "LangGraph"}]
Plan: Based on search results, LangGraph allows building stateful multi-actor applications with LLMs. I need to calculate something simple based on the result length.
#E2 = another_server/calculate[{"operand1": #E1, "operand2": 10, "operation": "add"}]
Plan: Calculation done. I can now provide the final answer including the search result and the calculation.
""" 
        # Note: The fake calculation result depends on the fake search result length + 10
        fake_search_result = "LangGraph is a library for building stateful, multi-actor applications with LLMs."
        # Calculation: len(fake_search_result) + 10 -> 86 + 10 = 96
        fake_calc_result = 96 
        # Solver will be called after worker returns the calculation result
        fake_final_answer = f"LangGraph is a library for building stateful, multi-actor applications with LLMs (found via search, Evidence 1). A calculation based on this resulted in {fake_calc_result} (Evidence 2)."

        # Mock the LLM responses for Planner and Solver
        fake_llm = FakeListLLM(responses=[fake_plan_output, fake_final_answer])

        # Tool execution uses the mock executor defined in executor.py via the registry

        # --- ---

        try:
            print("Getting agent executor with FakeLLM...")
            # Pass the fake LLM to the executor factory
            agent_executor = get_agent_executor(llm=fake_llm, force_reload=True) # Force reload for example

            print("\nInvoking agent graph...")
            initial_state = {"input_query": "What is LangGraph and calculate something based on result length?"}
            # Ensure the input state has all keys expected by the graph, even if empty initially
            initial_state.setdefault('plan', [])
            initial_state.setdefault('current_step_index', 0)
            initial_state.setdefault('evidence', [])
            initial_state.setdefault('final_answer', None)
            initial_state.setdefault('error', None)

            final_state = await agent_executor.ainvoke(initial_state)

            print("\n--- Final State --- ")
            import json
            # Print relevant parts of the final state, formatted nicely
            print(json.dumps({
                "input_query": final_state.get('input_query'),
                "plan_steps": len(final_state.get('plan', [])),
                "evidence_collected": len(final_state.get('evidence', [])),
                # Optionally include evidence details if short
                # "evidence": final_state.get('evidence', []),
                "final_answer": final_state.get('final_answer'),
                "error": final_state.get('error')
            }, indent=2))

            # Assertions
            assert final_state.get('final_answer') == fake_final_answer, f"Expected: {fake_final_answer}, Got: {final_state.get('final_answer')}"
            assert len(final_state.get('evidence', [])) == 2, f"Expected 2 pieces of evidence, got {len(final_state.get('evidence', []))}"
            assert final_state['evidence'][0]['tool_result']['output'] == fake_search_result
            assert final_state['evidence'][1]['tool_result']['output'] == fake_calc_result
            assert final_state.get('error') is None

            print("\nGraph example finished successfully.")

        except Exception as e:
            print(f"\nAn error occurred during graph example: {e}")
            # Print traceback for easier debugging
            import traceback
            traceback.print_exc()


    asyncio.run(run_graph_example()) 