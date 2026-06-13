
# ====================================================================
# INGESTION SCRIPT (Generalized for API use)
# ====================================================================
import os
import json
import hashlib
from typing import List
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_community.storage import RedisStore
from langchain_classic.storage import EncoderBackedStore
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

load_dotenv(override=True)

def ingest_documents(documents: List[Document]):
    """
    Generalized ingestion function. 
    Accepts a list of LangChain Document objects, making it easy to integrate with FastAPI endpoints.
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

if __name__ == "__main__":
    # Example usage: This block simulates what your FastAPI endpoint will do.
    # It handles loading the files and then passes the loaded documents to the pipeline.
    from langchain_community.document_loaders import JSONLoader
    
    huge_file = "corpus.jsonl"
    small_file = "small_corpus.jsonl"
    lines_to_keep = 50

    if os.path.exists(huge_file):
        print(f"Extracting first {lines_to_keep} lines from {huge_file}...")
        with open(huge_file, "r", encoding="utf-8") as infile, \
             open(small_file, "w", encoding="utf-8") as outfile:
            for i, line in enumerate(infile):
                if i >= lines_to_keep:
                    break
                outfile.write(line)

        loader = JSONLoader(
            file_path=small_file,
            jq_schema=".text",
            json_lines=True
        )
        loaded_docs = loader.load()
        
        # Call the generalized function
        ingest_documents(loaded_docs)
    else:
        print(f"File {huge_file} not found. Please provide valid documents to ingest.")
        
        
# from fastapi import FastAPI, UploadFile, File
# from langchain_community.document_loaders import PyPDFLoader
# import shutil

# app = FastAPI()

# @app.post("/ingest/")
# async def ingest_endpoint(file: UploadFile = File(...)):
#     # 1. Save uploaded file temporarily
#     file_path = f"temp_{file.filename}"
#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
    
#     # 2. Use the right LangChain loader based on file extension
#     loader = PyPDFLoader(file_path)
#     documents = loader.load()
    
#     # 3. Pass directly to your newly generalized ingestion function!
#     ingest_documents(documents)
    
#     return {"message": f"Successfully ingested {len(documents)} documents!"}