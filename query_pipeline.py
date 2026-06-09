# ====================================================================
# QUERY SCRIPT (Run Multiple Times)
# ====================================================================
import os
import json
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_openrouter import ChatOpenRouter
from pinecone import Pinecone
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Parent-Child
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_community.storage import RedisStore
from langchain_classic.storage import EncoderBackedStore
from langchain_core.documents import Document

# Reranker & Compression
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.document_compressors import LLMLinguaCompressor
from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline

# Reliability
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_core.rate_limiters import InMemoryRateLimiter

load_dotenv(override=True)

# 1. LOAD PERSISTED STORES
print("Connecting to Vector DB and Document Store...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5", 
    model_kwargs={'device': 'cpu'}, 
    encode_kwargs={'normalize_embeddings': True}
)

index_name = "gov-docs-hierarchical-test-2" 
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
print(f"Connecting to Redis Document Store at {redis_url}...")

# Base Redis store handles raw bytes
redis_store = RedisStore(redis_url=redis_url, namespace="gov-docs-parents")

# EncoderBackedStore wraps Redis so it can serialize LangChain Document objects
store = EncoderBackedStore(
    store=redis_store,
    key_encoder=lambda x: x,
    value_serializer=lambda doc: json.dumps(doc.to_json()).encode("utf-8"),
    value_deserializer=lambda b: Document(**json.loads(b.decode("utf-8")).get("kwargs", {}))
)

# Initialize splitters (Required by Pydantic validation even during retrieval)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)

parent_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 10} # Get top 10 children
)

# 2. RERANKER & LLMLINGUA PIPELINE
print("Initializing Compression Pipeline...")
cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)

llm_lingua_compressor = LLMLinguaCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    device_map="cpu",
    target_token=300
)

pipeline_compressor = DocumentCompressorPipeline(
    transformers=[reranker, llm_lingua_compressor]
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor, 
    base_retriever=parent_retriever
)

# 3. LLM CONFIGURATION
print("Configuring LLM...")
rate_limiter = InMemoryRateLimiter(requests_per_second=2, check_every_n_seconds=0.1, max_bucket_size=10)

primary_llm = ChatOpenRouter(
    model="openrouter/owl-alpha", 
    temperature=0.1,
    max_retries=3,               # Automatically retry API errors 3 times
    rate_limiter=rate_limiter    # Throttle requests seamlessly
)
fallback_llm = ChatOpenRouter(
    model="nex-agi/nex-n2-pro:free",       # Can be swapped for AzureOpenAI, Gemini, etc.
    temperature=0.1,
    max_retries=3,
    rate_limiter=rate_limiter
)
llm = primary_llm.with_fallbacks([fallback_llm])

# 4. QUERY FUNCTION
SYSTEM_PROMPT_TEMPLATE = """
You are a knowledgeable, strict assistant representing the details from government documents.
You are chatting with a user about government policies.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=15), reraise=True)
def robust_retrieve(question: str):
    return compression_retriever.invoke(question)

def answer_question_with_reranking(question: str):
    print(f"\n[?] Question: {question}")
    
    try:
        docs = robust_retrieve(question)
        print(f"[*] Retrieved & Compressed down to {len(docs)} highly relevant documents.")
    except Exception as e:
        print(f"[!] Retrieval failed: {e}")
        return
    
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt), 
            HumanMessage(content=question)
        ])
        print("\n[+] Answer:")
        print(response.content)
        return response.content
    except Exception as e:
        print(f"[!] Generation failed: {e}")

if __name__ == "__main__":
    test_question = "How did the introduction of the pregnancy checkbox on death certificates both improve and complicate the identification of maternal deaths, and what specific actions did CDC take between 2016 and 2020 to address the resulting data quality concerns before resuming publication of maternal mortality statistics?"
    answer_question_with_reranking(test_question)