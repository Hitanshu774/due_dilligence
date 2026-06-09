# ====================================================================
# INGESTION SCRIPT (Run Once)
# ====================================================================
import os
import json
from dotenv import load_dotenv

from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_community.storage import RedisStore
from langchain_classic.storage import EncoderBackedStore
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

load_dotenv(override=True)

def ingest_documents():
    # 1. DATA PREPARATION
    huge_file = "corpus.jsonl"
    small_file = "small_corpus.jsonl"
    lines_to_keep = 50

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
    documents = loader.load()
    print(f"Loaded {len(documents)} documents to ingest.")

    # 2. EMBEDDINGS & PINECONE SETUP
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

    # 3. PARENT-CHILD CHUNKING
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

    print("Setting up ParentDocumentRetriever and uploading to Pinecone & Redis...")
    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )

    # This is the expensive step: Embeds and uploads to Pinecone & Redis
    parent_retriever.add_documents(documents)
    print("✅ Vectors uploaded to Pinecone and Parents saved to Redis successfully!")
    
    print("✅ Ingestion complete! You can now run query.py")

if __name__ == "__main__":
    ingest_documents() 