import os
import json
import hashlib
import shutil
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import redis

# FastAPI Imports
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# LangChain / RAG Imports
from langchain_community.document_loaders import PyPDFLoader, JSONLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.storage import RedisStore
from langchain_classic.storage import EncoderBackedStore
from langchain_core.documents import Document, BaseDocumentCompressor
from langchain_core.callbacks import Callbacks
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker, DocumentCompressorPipeline
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_openrouter import ChatOpenRouter
from langchain_core.rate_limiters import InMemoryRateLimiter
from llmlingua import PromptCompressor
from pinecone import Pinecone, ServerlessSpec
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv(override=True)

# ====================================================================
# CUSTOM CLASSES & WRAPPERS
# ====================================================================
class LLMLingua2Compressor(BaseDocumentCompressor):
    target_token: int = 600
    compressor: Any = None
    
    class Config: arbitrary_types_allowed = True
        
    def compress_documents(self, documents: List[Document], query: str, callbacks: Optional[Callbacks] = None) -> List[Document]:
        if not documents: return []
        contexts = [doc.page_content for doc in documents]
        compressed = self.compressor.compress_prompt(context=contexts, instruction="", question=query, target_token=self.target_token)
        return [Document(page_content=compressed["compressed_prompt"])]

class MetadataFilterCompressor(BaseDocumentCompressor):
    metadata_filter: dict = {}
    def compress_documents(self, documents: List[Document], query: str, callbacks: Optional[Callbacks] = None) -> List[Document]:
        if not self.metadata_filter: return documents
        return [doc for doc in documents if all(doc.metadata.get(k) == v for k, v in self.metadata_filter.items())]

# ====================================================================
# GLOBAL APP STATE
# ====================================================================
# We store our initialized RAG components here so they aren't rebuilt on every request
class AppState:
    vectorstore: PineconeVectorStore = None
    store: EncoderBackedStore = None
    parent_retriever: ParentDocumentRetriever = None
    compression_retriever: ContextualCompressionRetriever = None
    metadata_filter_compressor: MetadataFilterCompressor = None
    llm: Any = None

state = AppState()

# ====================================================================
# API LIFESPAN (Startup & Shutdown)
# ====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes all heavy ML models and DB connections exactly once on startup."""
    print("🚀 Starting up RAG API & initializing models...")
    
    # 1. Embeddings & Pinecone
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "gov-docs-hierarchical-prod"
    if not pc.has_index(index_name):
        pc.create_index(name=index_name, dimension=384, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    state.vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    # 2. Redis Store
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = redis.Redis.from_url(redis_url, socket_timeout=120, socket_connect_timeout=120, retry_on_timeout=True)
    redis_store = RedisStore(client=redis_client, namespace="gov-docs-parents")
    state.store = EncoderBackedStore(
        store=redis_store, key_encoder=lambda x: x,
        value_serializer=lambda doc: json.dumps(doc.to_json()).encode("utf-8"),
        value_deserializer=lambda b: Document(**json.loads(b.decode("utf-8")).get("kwargs", {}))
    )

    # 3. Parent Retriever
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    state.parent_retriever = ParentDocumentRetriever(
        vectorstore=state.vectorstore, docstore=state.store,
        child_splitter=child_splitter, parent_splitter=parent_splitter, search_kwargs={"k": 10}
    )

    # 4. Hybrid Retriever (BM25 + Pinecone)
    all_keys = list(state.store.yield_keys())
    all_docs = [doc for doc in state.store.mget(all_keys) if doc is not None]
    
    if all_docs:
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = 10
        base_retriever = EnsembleRetriever(retrievers=[bm25_retriever, state.parent_retriever], weights=[0.3, 0.7])
    else:
        base_retriever = state.parent_retriever

    # 5. Compression Pipeline
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=5)
    llmlingua2_engine = PromptCompressor(model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank", use_llmlingua2=True, device_map="cpu")
    llm_lingua_compressor = LLMLingua2Compressor(target_token=600, compressor=llmlingua2_engine)
    state.metadata_filter_compressor = MetadataFilterCompressor()
    
    pipeline_compressor = DocumentCompressorPipeline(transformers=[state.metadata_filter_compressor, reranker, llm_lingua_compressor])
    state.compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=base_retriever)

    # 6. LLM
    rate_limiter = InMemoryRateLimiter(requests_per_second=2, check_every_n_seconds=0.1, max_bucket_size=10)
    primary_llm = ChatOpenRouter(model="openrouter/owl-alpha", temperature=0.1, max_retries=3, rate_limiter=rate_limiter)
    fallback_llm = ChatOpenRouter(model="nex-agi/nex-n2-pro:free", temperature=0.1, max_retries=3, rate_limiter=rate_limiter)
    state.llm = primary_llm.with_fallbacks([fallback_llm])

    print("✅ RAG API is fully ready!")
    yield
    print("🛑 Shutting down API...")

app = FastAPI(title="RAG Production API", lifespan=lifespan)

# ====================================================================
# BACKGROUND INGESTION TASK
# ====================================================================
def background_ingest_task(file_path: str, filename: str, custom_metadata: dict):
    """Runs asynchronously to prevent blocking the HTTP response."""
    try:
        if filename.endswith(".pdf"): loader = PyPDFLoader(file_path)
        elif filename.endswith(".jsonl"): loader = JSONLoader(file_path=file_path, jq_schema=".text", json_lines=True)
        elif filename.endswith(".txt"): loader = TextLoader(file_path)
        else: return
            
        documents = loader.load()
        for doc in documents: doc.metadata.update(custom_metadata)
        
        # Manually split to manage IDs
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        parent_docs = parent_splitter.split_documents(documents)
        
        unique_parents = {}
        for doc in parent_docs:
            doc_string = doc.page_content + json.dumps(doc.metadata, sort_keys=True)
            doc_id = hashlib.md5(doc_string.encode("utf-8")).hexdigest()
            unique_parents[doc_id] = doc
            
        parent_ids = list(unique_parents.keys())
        parent_docs = list(unique_parents.values())

        existing_records = state.store.mget(parent_ids)
        new_parent_docs = [pdoc for pdoc, record in zip(parent_docs, existing_records) if record is None]
        new_parent_ids = [pid for pid, record in zip(parent_ids, existing_records) if record is None]

        if new_parent_docs:
            state.parent_retriever.add_documents(new_parent_docs, ids=new_parent_ids)
            print(f"✅ Background Ingestion Complete: {len(new_parent_docs)} new chunks added.")
            
            # NOTE: In a true production app, we would trigger an update to the BM25 index here.
    except Exception as e:
        print(f"❌ Background Ingestion Failed: {str(e)}")
    finally:
        if os.path.exists(file_path): os.remove(file_path)

# ====================================================================
# API ROUTES
# ====================================================================
@app.get("/")
async def root():
    return HTMLResponse("<h1>RAG API Running</h1><p>Send POST requests to <code>/query</code> or visit <code>/docs</code></p>")

@app.post("/ingest/")
async def ingest_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    metadata_keys: List[str] = Form([]),
    metadata_values: List[str] = Form([])
):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    custom_metadata = {}
    for k, v in zip(metadata_keys, metadata_values):
        key, val = k.strip(), v.strip()
        if key and val:
            if val.isdigit(): val = int(val)
            custom_metadata[key] = val

    # Delegate the heavy lifting to the background task!
    background_tasks.add_task(background_ingest_task, file_path, file.filename, custom_metadata)
    
    return {"message": "Document accepted. Ingestion is processing in the background.", "metadata_applied": custom_metadata}

class QueryRequest(BaseModel):
    question: str
    filters: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    answer: str
    source_count: int

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    if request.filters:
        state.parent_retriever.search_kwargs["filter"] = request.filters
        state.metadata_filter_compressor.metadata_filter = request.filters
    else:
        state.parent_retriever.search_kwargs.pop("filter", None)
        state.metadata_filter_compressor.metadata_filter = {}

    try:
        docs = state.compression_retriever.invoke(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")
        
    if not docs:
        return {"answer": "I could not find any relevant documents matching those constraints.", "source_count": 0}

    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = f"You are a knowledgeable, strict assistant representing government documents.\nContext:\n{context}"
    
    try:
        response = state.llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=request.question)])
        return {"answer": response.content, "source_count": len(docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Generation failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)