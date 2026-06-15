import os
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openrouter import ChatOpenRouter  # Using OpenRouter based on your previous config
# from langchain_openai import ChatOpenAI # Uncomment if using direct OpenAI

from .state import AgentState

load_dotenv()

# ====================================================================
# 1. SCHEMAS
# ====================================================================
class ResearchPlan(BaseModel):
    """Schema for the Planner Agent to format its output."""
    internal_search_query: str = Field(
        ..., 
        description="Search query to send to the internal RAG database for private policies/data."
    )
    external_search_query: str = Field(
        ..., 
        description="Search query to execute on the public web for recent news/context."
    )
    reasoning: str = Field(
        ..., 
        description="Brief explanation of why these queries were chosen."
    )

# ====================================================================
# 2. PLANNER NODE (Agent 1)
# ====================================================================
def planner_node(state: AgentState):
    """
    Decomposes the complex user query into specific internal and external search queries.
    """
    query = state.get("user_query", "")
    print(f"\n[Agent: Planner] Analyzing query: '{query}'")
    
    # Initialize the LLM
    llm = ChatOpenRouter(model="openrouter/owl-alpha", temperature=0.1) 
    structured_llm = llm.with_structured_output(ResearchPlan)
    
    system_prompt = """
    You are a lead due-diligence research planner.
    Your job is to take a complex user query and break it down into two optimized search queries:
    1. An internal query for our secure government document database.
    2. An external query for the public web.
    Extract the core entities, intent, and be precise.
    """
    
    # Invoke the LLM to get a structured plan
    try:
        plan: ResearchPlan = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ])
        
        print(f"  -> Internal Query: {plan.internal_search_query}")
        print(f"  -> External Query: {plan.external_search_query}")
        
        # Write the plan back to the global state
        return {"plan": plan.model_dump()}
        
    except Exception as e:
        print(f"[!] Planner Node Failed: {e}")
        return {"plan": {"error": str(e)}}


# ====================================================================
# 3. INTERNAL RETRIEVAL NODE (Agent 2)
# ====================================================================
def internal_retrieval_node(state: AgentState):
    """
    Reads the plan, takes the internal query, and calls our local RAG microservice.
    """
    plan = state.get("plan", {})
    internal_query = plan.get("internal_search_query", "")
    
    print(f"\n[Agent: Internal Retrieval] Searching database for: '{internal_query}'")
    
    if not internal_query:
        return {"internal_evidence": "No internal query provided by planner."}
    
    # URL of the local FastAPI RAG Microservice we just started
    RAG_API_URL = "http://localhost:8000/query"
    
    payload = {
        "question": internal_query,
        # "filters": {"agency": "CDC"} # Optional: We could let the planner output metadata filters too!
    }
    
    try:
        # Ping the microservice
        response = requests.post(RAG_API_URL, json=payload, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            evidence = data.get("answer", "No answer generated.")
            source_count = data.get("source_count", 0)
            
            print(f"  -> Found internal evidence from {source_count} sources.")
            
            # Write the findings back to the global state
            return {"internal_evidence": evidence}
        else:
            error_msg = f"API Error {response.status_code}: {response.text}"
            print(f"  -> [!] {error_msg}")
            return {"internal_evidence": error_msg}
            
    except requests.exceptions.RequestException as e:
        print(f"  -> [!] Microservice connection failed. Is uvicorn running? Error: {e}")
        return {"internal_evidence": "Failed to connect to internal RAG database."}