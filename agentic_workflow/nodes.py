import os
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openrouter import ChatOpenRouter  # Using OpenRouter based on your previous config
# from langchain_openai import ChatOpenAI # Uncomment if using direct OpenAI
import requests

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
    Takes the user query and breaks it down into specific sub-queries
    for the internal and external retrieval agents.
    """
    query = state["user_query"]
    print(f"\n[Agent: Planner] Analyzing query: '{query}'")
    
    try:
        # Initialize our LLM (Adjust the model name to your OpenRouter preference if needed)
        llm = ChatOpenRouter(model="nvidia/nemotron-3-ultra-550b-a55b:free", temperature=0.1) 
        
        # Initialize the universal JSON parser
        parser = PydanticOutputParser(pydantic_object=ResearchPlan)
        
        system_prompt = f"""
        You are a lead due-diligence research planner.
        Your job is to take a complex user query and break it down into two highly optimized search queries:
        1. An internal query for our secure RAG document database.
        2. An external query for the public web.
        Be precise and focus on extracting the core entities and intent.
        
        {parser.get_format_instructions()}
        """
        
        # We invoke the standard LLM (No tool_choice required)
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ])
        
        # Parse the raw text output into our Pydantic object
        plan = parser.invoke(response)
        
        print(f"  -> Internal Query Generated: {plan.internal_search_query}")
        print(f"  -> External Query Generated: {plan.external_search_query}")
        print(f"  -> Reasoning: {plan.reasoning}")
        
        return {"plan": plan.model_dump()}
        
    except Exception as e:
        print(f"[!] Planner Node Failed: {e}")
        # Safe Fallback: If it fails, default to passing the raw user query to the search agents
        print("  -> Using raw query as fallback.")
        return {"plan": {
            "internal_search_query": query,
            "external_search_query": query,
            "reasoning": "Fallback activated due to parsing error."
        }}


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
        response = requests.post(RAG_API_URL, json=payload, timeout=180)
        
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
    
# ====================================================================
# 4. EXTERNAL RETRIEVAL NODE (Agent 3)
# ====================================================================
def external_retrieval_node(state: AgentState):
    """
    Reads the plan, takes the external query, and calls the Tavily Search API.
    """
    from langchain_community.tools.tavily_search import TavilySearchResults   
    
    plan = state.get("plan", {})
    external_query = plan.get("external_search_query", "")
    
    print(f"\n[Agent: External Retrieval] Searching the web for: '{external_query}'")
    
    if not external_query:
        return {"external_evidence": "No external query provided by planner."}
    
    try:
        # Initialize Tavily Search (Automatically reads TAVILY_API_KEY from .env)
        # We limit max_results to 3 to avoid overflowing the LLM's context window
        search_tool = TavilySearchResults(max_results=3)
        
        # Execute the search
        raw_results = search_tool.invoke({"query": external_query})
        
        # Format the output into a readable string for the downstream LLM agents
        evidence = ""
        for i, hit in enumerate(raw_results):
            url = hit.get('url', 'Unknown URL')
            content = hit.get('content', 'No content')
            evidence += f"--- Web Source {i+1} ---\nURL: {url}\nContent: {content}\n\n"
        
        print(f"  -> Successfully retrieved top {len(raw_results)} web sources.")
        
        # Write the web findings back to the global state
        return {"external_evidence": evidence}
        
    except Exception as e:
        error_msg = f"Tavily Search API Error: {str(e)}"
        print(f"  -> [!] {error_msg}")
        return {"external_evidence": error_msg}
    
# ====================================================================
# 5. CONTRADICTION DETECTION NODE (Agent 4)
# ====================================================================
def contradiction_node(state: AgentState):
    """
    Analyzes internal and external evidence for discrepancies or contradictions.
    """
    internal = state.get("internal_evidence", "No internal evidence retrieved.")
    external = state.get("external_evidence", "No external evidence retrieved.")
    
    print("\n[Agent: Contradiction Checker] Analyzing evidence for inconsistencies...")
    
    # Initialize the LLM (Using a smart model with low temperature for logical reasoning)
    llm = ChatOpenRouter(model="nvidia/nemotron-3.5-content-safety:free", temperature=0.1)
    
    system_prompt = """
    You are a meticulous due-diligence verification analyst.
    Your task is to compare two sets of information: 'Internal Evidence' (from proprietary documents) and 'External Evidence' (from public web searches).
    
    Analyze them and provide a brief report highlighting:
    1. Agreements: Where do the sources align?
    2. Discrepancies/Contradictions: Where do they disagree or present conflicting facts?
    
    If there are no contradictions, explicitly state that the sources align.
    Do NOT answer the user's original question. ONLY report on the relationship between the two evidence sets.
    Keep your analysis concise, structured, and factual.
    """
    
    user_prompt = f"--- INTERNAL EVIDENCE ---\n{internal}\n\n--- EXTERNAL EVIDENCE ---\n{external}"
    
    try:
        # Invoke the LLM to perform the comparison
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        print("  -> Contradiction analysis complete.")
        
        # Save the analysis to the global state
        return {"contradictions": response.content}
    
    except Exception as e:
        error_msg = f"Error during contradiction analysis: {str(e)}"
        print(f"  -> [!] {error_msg}")
        return {"contradictions": error_msg}

# ====================================================================
# 6. MEMO WRITER NODE (Agent 5)
# ====================================================================
def memo_writer_node(state: AgentState):
    """
    Synthesizes all evidence and the contradiction report into a final due-diligence memo.
    """
    query = state.get("user_query", "No query provided.")
    internal = state.get("internal_evidence", "None.")
    external = state.get("external_evidence", "None.")
    contradictions = state.get("contradictions", "None detected.")
    
    print("\n[Agent: Memo Writer] Drafting the final due-diligence report...")
    
    # Initialize the LLM (Using a slightly higher temperature for natural language generation)
    llm = ChatOpenRouter(model="nvidia/nemotron-3-super-120b-a12b:free", temperature=0.2)
    
    system_prompt = """
    You are a Senior Due-Diligence Analyst. 
    Your task is to write a final, comprehensive analytical memo answering the user's query.
    
    You must use ONLY the provided Internal Evidence, External Evidence, and Contradiction Report.
    
    Structure your memo professionally using Markdown:
    1. **Executive Summary:** A brief, direct answer to the query.
    2. **Internal Findings:** Insights strictly from the internal documents.
    3. **External Context:** Relevant public/web information.
    4. **Discrepancy Analysis:** A summary of any contradictions or alignments between internal and external sources.
    5. **Conclusion & Recommendations:** Final synthesized thoughts.
    
    Crucial Rule: You MUST include inline citations. 
    - When citing internal docs, use [Internal RAG].
    - When citing web sources, cite the URL or Source Name, e.g., [Web Source 1].
    If evidence is lacking to fully answer the query, clearly state what is missing.
    """
    
    user_prompt = f"""
    --- USER QUERY ---
    {query}
    
    --- INTERNAL EVIDENCE ---
    {internal}
    
    --- EXTERNAL EVIDENCE ---
    {external}
    
    --- CONTRADICTION / ALIGNMENT REPORT ---
    {contradictions}
    """
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        print("  -> Final memo successfully generated.")
        
        # Save the final memo to the state and add the final AI message to the chat history
        from langchain_core.messages import AIMessage
        return {
            "final_memo": response.content,
            "messages": [AIMessage(content=response.content)]
        }
        
    except Exception as e:
        error_msg = f"Error generating final memo: {str(e)}"
        print(f"  -> [!] {error_msg}")
        return {"final_memo": error_msg}