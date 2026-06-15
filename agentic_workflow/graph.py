# Connecting nodes with edges
import os
from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .nodes import (
    planner_node,
    internal_retrieval_node,
    external_retrieval_node,
    contradiction_node,
    memo_writer_node
)

def build_research_graph():
    """
    Constructs the LangGraph state machine for the due-diligence workflow.
    """
    print("[System] Initializing LangGraph Workflow...")
    
    # 1. Initialize the StateGraph with our custom AgentState
    workflow = StateGraph(AgentState)
    
    # 2. Add all the nodes (The "Workers")
    workflow.add_node("planner", planner_node)
    workflow.add_node("internal_retrieval", internal_retrieval_node)
    workflow.add_node("external_retrieval", external_retrieval_node)
    workflow.add_node("contradiction_checker", contradiction_node)
    workflow.add_node("memo_writer", memo_writer_node)
    
    # 3. Define the Edges (The "Flow")
    
    # Start -> Planner
    workflow.add_edge(START, "planner")
    
    # LangGraph uses a concept called "Supersteps" to handle parallel execution automatically. 
    # If we draw an edge from the Planner to both retrieval agents, LangGraph will fire them off 
    # at the exact same time. Then, if we point both of their outputs to the Contradiction Checker, 
    # LangGraph will smartly wait until both are finished, merge their data into the state, 
    # and then trigger the Contradiction Checker.
    
    # FAN-OUT: Planner -> Both Retrieval Agents (These will now run in PARALLEL)
    workflow.add_edge("planner", "internal_retrieval")
    workflow.add_edge("planner", "external_retrieval")
    
    # FAN-IN: Both Retrieval Agents -> Contradiction Checker
    # LangGraph automatically waits for BOTH parallel nodes to finish before moving on!
    workflow.add_edge("internal_retrieval", "contradiction_checker")
    workflow.add_edge("external_retrieval", "contradiction_checker")
    
    # Contradiction Checker -> Memo Writer
    workflow.add_edge("contradiction_checker", "memo_writer")
    
    # Memo Writer -> END
    workflow.add_edge("memo_writer", END)
    
    # 4. Compile the graph
    # (In a production app, you can pass a 'checkpointer' here to add memory/pausing!)
    compiled_graph = workflow.compile()
    
    print("[System] Graph compiled successfully!")
    return compiled_graph

# Instantiate the graph globally so it can be imported into your main app
app_graph = build_research_graph()