# from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openrouter import ChatOpenRouter
from due_dilligence.agentic_workflow.state import DueDiligenceState, PlannerOutput
from due_dilligence.agentic_workflow.tools import search_internal_documents, search_external_web

# Initialize our LLM (using gpt-4o-mini for speed and cost-effectiveness)
# Make sure OPENAI_API_KEY is in your .env file!
llm = ChatOpenRouter(model="nvidia/nemotron-3-super-120b-a12b:free")

# ==========================================
# 1. PLANNER NODE
# ==========================================
def plan_node(state: DueDiligenceState):
    """Uses LLM to break the user query into structured tasks."""
    query = state.get("user_query", "")
    
    # FIX: Use PydanticOutputParser instead of .with_structured_output 
    # This is much more resilient to API formatting quirks and safety proxies.
    parser = PydanticOutputParser(pydantic_object=PlannerOutput)
    
    # FIX: Softened prompt to bypass "Unauthorized Advice" safety filters
    sys_prompt = f"""You are a helpful research assistant organizing a search strategy. 
    Break the user's query into exactly 2 simple information-gathering tasks: one for internal documents, one for external web search.
    Do not provide any advice.
    
    {parser.get_format_instructions()}
    """
    
    # Invoke the standard LLM (no json_mode forcing needed)
    result = llm.invoke([
        SystemMessage(content=sys_prompt),
        HumanMessage(content=query)
    ])
    
    # Safely attempt to parse the output
    try:
        parsed_result = parser.invoke(result)
        # Convert Pydantic objects back to dicts to store in the state safely
        tasks = [task.model_dump() for task in parsed_result.tasks]
    except Exception as e:
        print(f"⚠️ Parsing failed (likely a safety filter or weird LLM output). Error: {e}")
        print("Fallback to default tasks.")
        # Fallback tasks to prevent the graph from crashing
        tasks = [
            {"task_id": "1", "description": f"Internal search for {query}", "source_type": "internal"},
            {"task_id": "2", "description": f"External search for {query}", "source_type": "external"}
        ]
        
    return {"plan": tasks}
# ==========================================
# 2. RETRIEVER NODE
# ==========================================
def retrieve_node(state: DueDiligenceState):
    """Executes the plan using tools. (No LLM needed here, just logic!)"""
    internal_results = []
    external_results = []
    
    for task in state.get("plan", []):
        if task["source_type"] == "internal":
            # Notice how we pass the LLM-generated description directly to the tool!
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
    """Uses LLM to check for contradictions between internal and external evidence."""
    internal = "\n".join(state.get("internal_evidence", []))
    external = "\n".join(state.get("external_evidence", []))
    
    # FIX: Softened prompt to bypass safety filters. Removed "risk analyst".
    prompt = f"""
    You are a text comparison assistant. Compare Text A (Internal Evidence) with Text B (External Evidence).
    Identify any factual differences or contradictions between the two texts. 
    If there are none, say "No major contradictions found."
    Do not provide financial advice.
    
    Text A (Internal):
    {internal}
    
    Text B (External):
    {external}
    """
    
    # FIX: Added try/except to prevent the app from freezing if the API hangs
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content
    except Exception as e:
        print(f"⚠️ Verifier LLM Error: {e}")
        content = f"Verification skipped due to API error: {e}"
    
    return {"contradictions": [content]}

# ==========================================
# 4. WRITER NODE
# ==========================================
def write_node(state: DueDiligenceState):
    """Synthesizes the final memo using the LLM."""
    internal = "\n- ".join(state.get("internal_evidence", []))
    external = "\n- ".join(state.get("external_evidence", []))
    issues = "\n- ".join(state.get("contradictions", []))
    
    prompt = f"""
    Write a professional Due Diligence Memo based on the following findings.
    Be concise, objective, and clearly state any risks.
    
    Internal Findings:
    {internal}
    
    External Findings:
    {external}
    
    Risk & Verification Analysis:
    {issues}
    """
    
    response = llm.invoke([
        SystemMessage(content="You are a meticulous financial/legal analyst writing a final report."),
        HumanMessage(content=prompt)
    ])
    
    return {"final_memo": response.content}