#Data structures (TypedDicts & Pydantic)

import operator
from typing import Annotated, List, TypedDict
from pydantic import BaseModel, Field

# ==========================================
# PYDANTIC MODELS (For LLM Structured Output)
# ==========================================
class SubTask(BaseModel):
    task_id: str = Field(description="Unique identifier for the task, e.g., 'task_1'")
    description: str = Field(description="Description of what needs to be researched")
    source_type: str = Field(description="Must be exactly 'internal' or 'external'")

class PlannerOutput(BaseModel):
    tasks: List[SubTask] = Field(description="List of sub-tasks to execute for the due diligence")

# ==========================================
# LANGGRAPH STATE (The Shared Memory)
# ==========================================
class DueDiligenceState(TypedDict):
    user_query: str
    plan: List[dict] # Storing the tasks as dicts
    internal_evidence: Annotated[List[str], operator.add]
    external_evidence: Annotated[List[str], operator.add]
    contradictions: Annotated[List[str], operator.add]
    final_memo: str
    revision_count: int