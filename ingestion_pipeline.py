import os
import json
import hashlib
import shutil
from typing import List
from dotenv import load_dotenv

# FastAPI Imports
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import uvicorn

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_community.storage import RedisStore
from langchain_classic.storage import EncoderBackedStore
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

load_dotenv(override=True)

# Create the FastAPI App
app = FastAPI(title="RAG Ingestion API")

# --- DIRECT HTML UPLOAD PAGE ---
@app.get("/")
async def root():
    """Serves a simple HTML UI for uploading files directly from the browser."""
    html_content = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>RAG Document Ingestion</title>
            <style>
                body { font-family: sans-serif; margin: 40px; display: flex; justify-content: center; background-color: #f4f4f9; }
                .container { background: white; border: 1px solid #ccc; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); max-width: 500px; width: 100%; }
                h2 { margin-top: 0; color: #333; }
                p { color: #666; margin-bottom: 20px; }
                input[type=file] { margin-bottom: 20px; width: 100%; }
                .btn { padding: 10px 15px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; width: 100%; }
                .btn:hover { background-color: #0056b3; }
            </style>
        </head>
        <body>
            <div class="container">
                <h2>RAG Pipeline Ingestion</h2>
                <p>Select a PDF file to chunk, embed, and ingest into Pinecone and Redis.</p>
                <form action="/ingest/" enctype="multipart/form-data" method="post">
                    <input name="file" type="file" accept=".pdf" required>
                    <button class="btn" type="submit">Upload & Ingest</button>
                </form>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# --- YOUR INGESTION LOGIC ---
def ingest_documents(documents: List[Document]):
    """
    Generalized ingestion function. 
    Accepts a list of LangChain Document objects.
    """
    if not documents:
        print("No documents provided for ingestion.")
        return

    print(f"Received {len(documents)} documents to ingest.")

    # 1. EMBEDDINGS & PINECONE SETUP
    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5", 
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "gov-docs-hierarchical-test-2" 

    if not pc.has_index(index_name):
        print(f"Creating new Pinecone index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=384, 
            metric="cosine", 
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    else:
        print(f"Index '{index_name}' already exists. We will add to it.")

    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )

    # 2. PARENT-CHILD CHUNKING & DEDUPLICATION
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    print(f"Connecting to Redis at {redis_url}...")
    
    # Base Redis store handles raw bytes
    redis_store = RedisStore(redis_url=redis_url, namespace="gov-docs-parents")
    
    # EncoderBackedStore wraps Redis so it can serialize LangChain Document objects
    store = EncoderBackedStore(
        store=redis_store,
        key_encoder=lambda x: x,
        value_serializer=lambda doc: json.dumps(doc.to_json()).encode("utf-8"),
        value_deserializer=lambda b: Document(**json.loads(b.decode("utf-8")).get("kwargs", {}))
    )
    
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)

    print("Setting up ParentDocumentRetriever...")
    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=None # Set to None because we pre-split below to manage IDs
    )

    print("Pre-splitting and hashing documents to prevent double-ingestion...")
    # A. Manually split into parents first
    parent_docs = parent_splitter.split_documents(documents)
    
    # B. Generate deterministic IDs based on content and metadata
    unique_parents = {}
    for doc in parent_docs:
        doc_string = doc.page_content + json.dumps(doc.metadata, sort_keys=True)
        doc_id = hashlib.md5(doc_string.encode("utf-8")).hexdigest()
        unique_parents[doc_id] = doc
        
    parent_ids = list(unique_parents.keys())
    parent_docs = list(unique_parents.values())

    # C. Check Redis for existing parent IDs
    print(f"Checking Redis to see if {len(parent_ids)} parent chunks already exist...")
    existing_records = store.mget(parent_ids)
    
    new_parent_docs = []
    new_parent_ids = []
    
    for pdoc, pid, existing_record in zip(parent_docs, parent_ids, existing_records):
        if existing_record is None:
            new_parent_docs.append(pdoc)
            new_parent_ids.append(pid)

    # D. Ingest ONLY the new documents
    if not new_parent_docs:
        print("✅ All documents are already ingested. Skipping to prevent duplication.")
    else:
        print(f"[*] Found {len(new_parent_docs)} new parent chunks to ingest. (Skipped {len(parent_docs) - len(new_parent_docs)} duplicates)")
        parent_retriever.add_documents(new_parent_docs, ids=new_parent_ids)
        print("✅ New vectors uploaded to Pinecone and Parents saved to Redis successfully!")
    
    print("✅ Ingestion complete!")

# --- FASTAPI FILE UPLOAD ENDPOINT ---
@app.post("/ingest/")
async def ingest_endpoint(file: UploadFile = File(...)):
    # 1. Save uploaded file temporarily
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 2. Use the right LangChain loader based on file extension
    try:
        if file.filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        else:
            return {"error": "Currently only PDF files are supported."}
            
        # 3. Pass directly to your newly generalized ingestion function!
        ingest_documents(documents)
        
        return {"message": f"Successfully ingested {len(documents)} document pages!"}
        
    finally:
        # 4. Clean up the temporary file so we don't fill up the disk
        if os.path.exists(file_path):
            os.remove(file_path)

# --- STARTUP COMMAND ---
if __name__ == "__main__":
    print("Starting FastAPI Ingestion Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)