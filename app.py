# Gradio UI & LangSmith 
import gradio as gr
import os
# from graph import app # Import the compiled graph we made in Phase 4

# ==========================================
# 0. LANGSMITH & API CONFIGURATION
# ==========================================
# In a real project, keep these in a .env file and use python-dotenv to load them!
os.environ["LANGCHAIN_TRACING_V2"] = "true" # This is the magic switch that turns on LangSmith!
os.environ["LANGCHAIN_API_KEY"] = "ls__your_langsmith_api_key_here" # Get this from smith.langchain.com
os.environ["LANGCHAIN_PROJECT"] = "Due_Diligence_Agent_v1" # Groups your traces under this project name

# Make sure to set your LLM API keys as well for the nodes to work
# os.environ["OPENAI_API_KEY"] = "sk-..." 

# ==========================================
# 1. EXECUTION FUNCTION
# ==========================================
def run_due_diligence(user_query: str):
    """
    This function initializes the state and runs the LangGraph app.
    We use .stream() to yield updates to the Gradio UI step-by-step.
    """
    
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
    yield output_log # Send initial status to UI
    
    # 2. Execute using .stream()
    try:
        # NOTE: If running the real graph, you would use:
        # for event in app.stream(initial_state, stream_mode="updates"):
        
        # MOCKING the stream for demonstration purposes
        mock_events = [
            {"Planner": {"plan": [{"task": "Search internal DB"}, {"task": "Search Web"}]}},
            {"Retriever": {"internal_evidence": ["Doc 1"], "external_evidence": ["Web 1"]}},
            {"Verifier": {"contradictions": ["No contradictions found."]}},
            {"Writer": {"final_memo": "Here is the final Due Diligence Report..."}}
        ]
        
        for event in mock_events: # Replace `mock_events` with `app.stream(...)`
            # event is a dict where the key is the Node name and value is the state update
            for node_name, state_update in event.items():
                output_log += f"✅ {node_name} Node Completed.\n"
                
                # If the writer finishes, we can show the final memo
                if node_name == "Writer":
                    output_log += "\n" + "="*40 + "\n"
                    output_log += f"📝 FINAL MEMO:\n{state_update['final_memo']}\n"
                
                # Yield the updated log to Gradio so the text box updates in real-time!
                yield output_log 
                
    except Exception as e:
        yield f"❌ Error occurred: {str(e)}"

# ==========================================
# 2. GRADIO UI SETUP
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🕵️‍♂️ AI Due-Diligence Agent (Powered by LangGraph & Traced by LangSmith)")
    gr.Markdown("Enter a company or query to begin the multi-agent research process.")
    
    with gr.Row():
        with gr.Column(scale=1):
            query_input = gr.Textbox(
                label="Research Query", 
                placeholder="e.g., Conduct due diligence on Target Company's Q3 compliance..."
            )
            submit_btn = gr.Button("Run Analysis", variant="primary")
            
        with gr.Column(scale=2):
            # This output box will fill up step-by-step thanks to yield!
            output_display = gr.Textbox(
                label="Agent Reasoning & Output", 
                lines=15, 
                interactive=False
            )
            
    # Connect the button to the function. 
    submit_btn.click(
        fn=run_due_diligence,
        inputs=[query_input],
        outputs=[output_display]
    )

if __name__ == "__main__":
    demo.launch()