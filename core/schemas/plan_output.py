from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ToolCall(BaseModel):
    tool_name: str = Field(..., description="Name of the tool to call, e.g., 'server_name/tool_name'.")
    arguments: Dict[str, Any] = Field(..., description="Arguments for the tool call as a JSON dictionary.")

class PlanStep(BaseModel):
    thought: str = Field(..., description="Reasoning process for why this step is needed.")
    tool_call: Optional[ToolCall] = Field(None, description="The tool call to execute for this step, if any.")
    expected_outcome: str = Field(..., description="Brief description of the expected information or result from this step.")

class PlanOutput(BaseModel):
    steps: List[PlanStep] = Field(..., description="List of sequential steps in the plan.") 