# The "Agents" (functions that modify state)
from state import DueDiligenceState
from tools import search_internal_documents, search_external_web

# ==========================================
# 1. PLANNER NODE
# ==========================================
def plan_node(state: DueDiligenceState):
    """Breaks the user query into tasks. (Mocked LLM behavior)"""
    query = state.get("user_query", "").lower()
    
    # Mock planning
    tasks = [
        {"task_id": "1", "description": f"Internal search for {query}", "source_type": "internal"},
        {"task_id": "2", "description": f"External search for {query}", "source_type": "external"}
    ]
    return {"plan": tasks}

# ==========================================
# 2. RETRIEVER NODE
# ==========================================
def retrieve_node(state: DueDiligenceState):
    """Executes the plan using tools."""
    internal_results = []
    external_results = []
    
    for task in state.get("plan", []):
        if task["source_type"] == "internal":
            result = search_internal_documents.invoke(task["description"])
            internal_results.append(result)
        else:
            result = search_external_web.invoke(task["description"])
            external_results.append(result)
            
    return {
        "internal_evidence": internal_results, 
        "external_evidence": external_results
    }

# ==========================================
# 3. VERIFIER NODE
# ==========================================
def verify_node(state: DueDiligenceState):
    """Checks for contradictions between internal and external evidence."""
    internal = " ".join(state.get("internal_evidence", []))
    external = " ".join(state.get("external_evidence", []))
    
    # Mock contradiction logic
    if "crisis" in external and "grew" in internal:
        contradiction = "Contradiction found: Internal says growing, external reports crisis."
    else:
        contradiction = "No major contradictions found."
        
    return {"contradictions": [contradiction]}

# ==========================================
# 4. WRITER NODE
# ==========================================
def write_node(state: DueDiligenceState):
    """Synthesizes the final memo."""
    internal = "\n- ".join(state.get("internal_evidence", []))
    external = "\n- ".join(state.get("external_evidence", []))
    issues = "\n- ".join(state.get("contradictions", []))
    
    memo = f"""### Due Diligence Memo
**Internal Findings:**
- {internal}

**External Findings:**
- {external}

**Risk & Verification:**
- {issues}
"""
    return {"final_memo": memo}