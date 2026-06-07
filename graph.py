# Connecting nodes with edges
from langgraph.graph import StateGraph, END
from state import DueDiligenceState
from nodes import plan_node, retrieve_node, verify_node, write_node

# 1. Initialize Graph
workflow = StateGraph(DueDiligenceState)

# 2. Add Nodes
workflow.add_node("Planner", plan_node)
workflow.add_node("Retriever", retrieve_node)
workflow.add_node("Verifier", verify_node)
workflow.add_node("Writer", write_node)

# 3. Entry Point
workflow.set_entry_point("Planner")

# 4. Add Edges (Linear Flow for now)
workflow.add_edge("Planner", "Retriever")
workflow.add_edge("Retriever", "Verifier")
workflow.add_edge("Verifier", "Writer")
workflow.add_edge("Writer", END)

# 5. Compile
compiled_graph = workflow.compile()