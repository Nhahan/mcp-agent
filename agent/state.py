# agent/state.py
from typing import TypedDict, List, Optional, Dict, Any, Union

# Tool-related data structures (might overlap/refine with tools/models.py later)
class ToolCall(TypedDict):
    """ Represents a planned tool call within the ReWOO plan. """
    tool_name: str # e.g., "web_search_placeholder/search"
    arguments: Dict[str, Any]

class ToolResult(TypedDict):
    """ Represents the output of a tool execution. """
    tool_name: str
    output: Any # The actual data returned by the tool
    error: Optional[str] # Any error message during execution

# ReWOO step-specific data structures
class PlanStep(TypedDict):
    """ Represents a single step in the reasoning plan. """
    step: int
    thought: str
    tool_call: str
    expected_outcome: str
    status: str # 'pending', 'in_progress', 'completed', 'failed'

class Evidence(TypedDict):
    """ Represents the evidence gathered from a tool execution, linked to a plan step. """
    step_index: int # The index of the plan step this evidence corresponds to (matches PlanStep.step_index)
    tool_result: ToolResult # The result of the tool execution
    processed_evidence: str # A summary or processed form of the tool output (potentially LLM generated)

# LangGraph State Definition
class ReWOOState(TypedDict):
    """
    Represents the state of the ReWOO agent workflow.
    """
    original_query: str
    plan: List[PlanStep]
    current_step_index: int
    tool_name: str | None
    tool_input: Dict[str, Any] | None
    tool_output: Any | None # Can be string, dict, or other types depending on the tool
    evidence: List[str] # List of collected evidence strings
    final_answer: str | None
    error_message: str | None
    max_retries: int
    current_retry: int
    # Additional fields for state management if needed
    # e.g., validation_errors: List[str] | None
    workflow_status: str # 'planning', 'tool_selection', 'tool_input_prep', 'tool_execution', 'evidence_collection', 'final_answer_gen', 'completed', 'failed' 