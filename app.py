import gradio as gr
import os
from dotenv import load_dotenv
from langsmith import traceable

# Load environment variables from .env file
load_dotenv()

# IMPORTANT: Import our actual compiled graph!
from graph import compiled_graph

# ==========================================
# 0. LANGSMITH & API CONFIGURATION
# ==========================================
# We no longer need to hardcode os.environ[] here because load_dotenv() 
# securely pulls them from the .env file!

# ==========================================
# 1. EXECUTION FUNCTION
# ==========================================

def run_due_diligence(user_query: str):
    
    # 1. Setup the initial state
    initial_state = {
        "user_query": user_query,
        "plan": [],
        "internal_evidence": [],
        "external_evidence": [],
        "contradictions": [],
        "final_memo": "",
        "revision_count": 0
    }
    
    output_log = f"🚀 Starting Due Diligence for: '{user_query}'\n"
    output_log += "-" * 40 + "\n"
    yield output_log 
    
    # 2. Execute the REAL graph using .stream()
    try:
        # We use stream_mode="updates" to yield each node's output as it finishes
        for event in compiled_graph.stream(initial_state, stream_mode="updates"):
            
            for node_name, state_update in event.items():
                output_log += f"✅ {node_name} Node Completed.\n"
                
                # If the Writer node runs, append the actual memo to the UI
                if node_name == "Writer":
                    output_log += "\n" + "="*40 + "\n"
                    output_log += f"{state_update['final_memo']}\n"
                
                yield output_log 
                
    except Exception as e:
        yield f"❌ Error occurred: {str(e)}"

# ==========================================
# 2. GRADIO UI SETUP
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🕵️‍♂️ AI Due-Diligence Agent (LangGraph + LangSmith)")
    gr.Markdown("Enter a query like 'financial' or 'compliance' to test the mock nodes.")
    
    with gr.Row():
        with gr.Column(scale=1):
            query_input = gr.Textbox(
                label="Research Query", 
                placeholder="Type 'financial' to see contradiction logic trigger..."
            )
            submit_btn = gr.Button("Run Analysis", variant="primary")
            
        with gr.Column(scale=2):
            output_display = gr.Textbox(
                label="Agent Reasoning & Output", 
                lines=15, 
                interactive=False
            )
            
    submit_btn.click(
        fn=run_due_diligence,
        inputs=[query_input],
        outputs=[output_display]
    )

if __name__ == "__main__":
    # Launch the Gradio app
    demo.launch()