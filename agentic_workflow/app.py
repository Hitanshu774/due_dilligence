import gradio as gr
from langchain_core.messages import HumanMessage
import sys
import os

# Ensure the parent directory is in the path so we can import the graph
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agentic_workflow.graph import app_graph

def generate_due_diligence_memo(user_query, history):
    """
    Takes the user's query from Gradio, passes it to the LangGraph orchestrator,
    and returns the final synthesized memo.
    """
    print(f"\n[UI] Received new query: {user_query}")
    
    # 1. Initialize the state exactly as LangGraph expects
    initial_state = {
        "user_query": user_query,
        "messages": [HumanMessage(content=user_query)]
    }
    
    try:
        # 2. Invoke the compiled graph (Synchronous execution)
        # This will trigger the Planner -> Retrievers -> Contradiction Checker -> Memo Writer
        print("[UI] Invoking Multi-Agent Graph...")
        result = app_graph.invoke(initial_state)
        
        # 3. Extract the final memo from the returned state
        final_memo = result.get("final_memo")
        
        if final_memo:
            print("[UI] Memo generated successfully!")
            return final_memo
        else:
            return "⚠️ The agents completed their run, but no final memo was generated. Check the terminal logs."
            
    except Exception as e:
        error_msg = f"❌ An error occurred during the multi-agent workflow: {str(e)}"
        print(error_msg)
        return error_msg

# ==========================================
# Build the Gradio UI
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🕵️‍♂️ Due Diligence & Research Assistant
        **A LangGraph Multi-Agent System**
        
        This assistant breaks down your query, searches internal government documents via a local RAG microservice, 
        conducts external web research, checks for contradictions, and synthesizes a final citation-backed memo.
        """
    )
    
    # Use ChatInterface for a clean, modern chat experience
    chat_interface = gr.ChatInterface(
        fn=generate_due_diligence_memo,
        chatbot=gr.Chatbot(height=600, show_copy_button=True),
        textbox=gr.Textbox(
            placeholder="e.g., What are the DOD rules for depot maintenance, and is there any recent news about violations?", 
            container=False, 
            scale=7
        ),
        theme="soft",
    )

if __name__ == "__main__":
    print("Starting Gradio Server...")
    # launch(share=True) gives you a public link to share with others!
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)