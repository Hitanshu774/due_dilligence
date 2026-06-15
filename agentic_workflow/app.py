import gradio as gr
from langchain_core.messages import HumanMessage
import sys
import os
import requests

# Ensure the parent directory is in the path so we can import the graph
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agentic_workflow.graph import app_graph

# ==========================================
# 1. CHAT / AGENT LOGIC
# ==========================================
def generate_due_diligence_memo(user_query, history):
    """
    Takes the user's query from Gradio, passes it to the LangGraph orchestrator,
    and returns the final synthesized memo.
    """
    print(f"\n[UI] Received new query: {user_query}")
    
    initial_state = {
        "user_query": user_query,
        "messages": [HumanMessage(content=user_query)]
    }
    
    try:
        print("[UI] Invoking Multi-Agent Graph...")
        result = app_graph.invoke(initial_state)
        
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
# 2. INGESTION UPLOAD LOGIC
# ==========================================
def upload_to_rag(file_paths, meta_key, meta_value):
    """
    Takes files from the Gradio UI and forwards them to the FastAPI ingestion microservice.
    """
    if not file_paths:
        return "⚠️ Please select at least one file to upload."
        
    url = "http://localhost:8000/ingest/"
    responses = []
    
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        print(f"[UI] Forwarding {filename} to Ingestion Microservice...")
        
        try:
            with open(file_path, "rb") as f:
                # Prepare the multipart form data
                files = {"file": (filename, f)}
                data = {}
                
                # Attach metadata if the user provided it
                if meta_key and meta_value:
                    data = {
                        "metadata_keys": [meta_key.strip()], 
                        "metadata_values": [meta_value.strip()]
                    }
                else:
                    data = {"metadata_keys": [], "metadata_values": []}
                    
                # Send to FastAPI!
                response = requests.post(url, files=files, data=data)
                
                if response.status_code == 200:
                    resp_json = response.json()
                    responses.append(f"✅ {filename}: {resp_json.get('message')} (Metadata: {resp_json.get('metadata_applied')})")
                else:
                    responses.append(f"❌ {filename}: Error {response.status_code} - {response.text}")
                    
        except requests.exceptions.ConnectionError:
            responses.append(f"❌ {filename}: Connection Error! Is your FastAPI RAG microservice running on port 8000?")
        except Exception as e:
            responses.append(f"❌ {filename}: Failed to upload. Error: {str(e)}")
            
    return "\n\n".join(responses)


# ==========================================
# 3. BUILD THE GRADIO UI
# ==========================================
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🕵️‍♂️ Due Diligence & Research Assistant
        **A LangGraph Multi-Agent System**
        """
    )
    
    with gr.Tabs():
        # --- TAB 1: The Chat Agent ---
        with gr.TabItem("💬 Agent Chat"):
            gr.Markdown("Ask complex questions. The agent will plan, retrieve internal/external data, check for contradictions, and write a memo.")
            chat_interface = gr.ChatInterface(
                fn=generate_due_diligence_memo,
                chatbot=gr.Chatbot(height=550),
                textbox=gr.Textbox(
                    placeholder="e.g., What are the DOD rules for depot maintenance...", 
                    container=False, 
                    scale=7
                )
            )
            
        # --- TAB 2: The Document Uploader ---
        with gr.TabItem("📁 Knowledge Base Ingestion"):
            gr.Markdown("Upload new documents directly into the local vector database. The agent will instantly be able to search them.")
            
            with gr.Row():
                with gr.Column(scale=2):
                    file_input = gr.File(
                        label="Select Documents (.pdf, .txt, .jsonl)", 
                        file_count="multiple",
                        type="filepath"
                    )
                with gr.Column(scale=1):
                    gr.Markdown("### Custom Metadata (Optional)")
                    gr.Markdown("Tag your documents so the agent can filter them later.")
                    meta_key = gr.Textbox(label="Metadata Key (e.g., agency)")
                    meta_value = gr.Textbox(label="Metadata Value (e.g., CDC)")
                    
                    upload_btn = gr.Button("🚀 Upload to Knowledge Base", variant="primary")
            
            upload_status = gr.Textbox(label="Upload Status Log", interactive=False, lines=5)
            
            # Wire up the button
            upload_btn.click(
                fn=upload_to_rag,
                inputs=[file_input, meta_key, meta_value],
                outputs=upload_status
            )

if __name__ == "__main__":
    print("Starting Unified Gradio Server...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)