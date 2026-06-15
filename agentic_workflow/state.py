#Data structures (TypedDicts & Pydantic)

import operator
from typing import Annotated, TypedDict

class AgentState(TypedDict):
    """
    This dictionary is the shared memory for all agents in the LangGraph workflow.
    Every agent will read from and write to these specific fields.
    """
    user_query: str
    plan: dict                 # Populated by the Planner Node
    internal_evidence: str     # Populated by the Internal Retrieval Node (RAG)
    external_evidence: str     # Populated by the External Retrieval Node (Web)
    contradictions: str        # Populated by the Contradiction-checking Node
    final_memo: str            # Populated by the Memo-Writing Node
    
    # Annotated with operator.add means messages get appended, not overwritten
    messages: Annotated[list, operator.add]