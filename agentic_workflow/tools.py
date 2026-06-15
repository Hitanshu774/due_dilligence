# Retrieval functions
from langchain_core.tools import tool

# ==========================================
# 1. INTERNAL RAG TOOL
# ==========================================
@tool
def search_internal_documents(query: str) -> str:
    """Search the internal company database, financial reports, and uploaded documents."""
    mock_db = {
        "financial": "Internal Q3 Report: Revenue grew by 15%, but cash flow is strained due to high CapEx.",
        "compliance": "Internal Audit (Dec 2023): No major compliance violations found.",
        "default": "No specific internal documents found for this query."
    }
    
    query_lower = query.lower()
    if "financial" in query_lower or "revenue" in query_lower or "cash" in query_lower:
        return mock_db["financial"]
    elif "compliance" in query_lower or "audit" in query_lower:
        return mock_db["compliance"]
    return mock_db["default"]

# ==========================================
# 2. EXTERNAL WEB SEARCH TOOL
# ==========================================
@tool
def search_external_web(query: str) -> str:
    """Search the external web for news, public records, and articles."""
    mock_web = {
        "financial": "Bloomberg article (Yesterday): 'Target Company facing severe cash flow crisis.'",
        "compliance": "Reuters breaking news: 'European regulators open investigation into Target Company.'",
        "default": "Web search returned general marketing materials."
    }
    
    query_lower = query.lower()
    if "financial" in query_lower or "cash" in query_lower or "revenue" in query_lower:
        return mock_web["financial"]
    elif "compliance" in query_lower or "investigat" in query_lower:
        return mock_web["compliance"]
    return mock_web["default"]