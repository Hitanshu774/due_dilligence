# ====================================================================
# 1. IMPORTS & SETUP
# ====================================================================
import os
import glob
from dotenv import load_dotenv

# Standard Langchain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_openrouter import ChatOpenRouter
from langchain_pinecone import PineconeVectorStore

# Reranker Imports
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

# Pinecone
from pinecone import Pinecone, ServerlessSpec

# =========================================
# NEW IMPORTS FOR RELIABILITY
# =========================================
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_core.rate_limiters import InMemoryRateLimiter

load_dotenv(override=True)


# ====================================================================
# 2. DATA LOADING & CHUNKING
# ====================================================================
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
print(f"Loaded {len(documents)} documents from small corpus")

textsplitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = textsplitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")


# ====================================================================
# 3. GENERATE EMBEDDINGS & PUSH TO PINECONE
# ====================================================================
print("Initializing CPU-friendly embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5", 
    model_kwargs={'device': 'cpu'}, 
    encode_kwargs={'normalize_embeddings': True}
)

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "gov-docs-index-small" 

if not pc.has_index(index_name):
    print(f"Creating new Pinecone index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=384, 
        metric="cosine", 
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

print(f"Pushing {len(chunks)} chunks to Pinecone index: '{index_name}'...")
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=index_name
)


# ====================================================================
# 4. RERANKER SETUP
# ====================================================================
print("Initializing Cross-Encoder Reranker...")
base_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
compressor = CrossEncoderReranker(model=cross_encoder, top_n=3)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=base_retriever
)


# ====================================================================
# 5. LLM CONFIGURATION (RATE LIMITING, RETRIES & FALLBACKS)
# ====================================================================
print("Configuring LLMs and Rate Limiter...")

# A. Internal Rate Limiter (e.g., 2 requests per second max)
rate_limiter = InMemoryRateLimiter(
    requests_per_second=2,
    check_every_n_seconds=0.1,
    max_bucket_size=10, 
)

# B. Primary Model with Built-in Langchain Retries & Rate Limiter
primary_llm = ChatOpenRouter(
    model="openrouter/owl-alpha", 
    temperature=0.1,
    max_retries=3,               # Automatically retry API errors 3 times
    rate_limiter=rate_limiter    # Throttle requests seamlessly
)

# C. Fallback Model
# If OpenRouter is down or hits hard limits, it will instantly reroute to OpenAI
fallback_llm = ChatOpenRouter(
    model="nex-agi/nex-n2-pro:free",       # Can be swapped for AzureOpenAI, Gemini, etc.
    temperature=0.1,
    max_retries=3,
    rate_limiter=rate_limiter
)

# D. Combine into a resilient LLM object
llm = primary_llm.with_fallbacks([fallback_llm])


# ====================================================================
# 6. PIPELINE EXECUTION (WITH TENACITY EXPONENTIAL BACKOFF)
# ====================================================================
SYSTEM_PROMPT_TEMPLATE = """
You are a knowledgeable, strict assistant representing the details from government documents.
You are chatting with a user about government policies.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

# Wrap the retriever in Tenacity to protect against Pinecone/Embedding API network errors
@retry(
    stop=stop_after_attempt(5),                                # Stop after 5 total attempts
    wait=wait_exponential(multiplier=1, min=2, max=15),        # Waits 2s, 4s, 8s, up to 15s
    reraise=True                                               # Reraise exception if all retries fail
)
def robust_retrieve(question: str):
    """Retrieves documents with exponential backoff on failure."""
    return compression_retriever.invoke(question)

def answer_question_with_reranking(question: str):
    print(f"\n[?] Question: {question}")
    
    # 1. Retrieve & Rerank (with Retry Logic)
    try:
        docs = robust_retrieve(question)
        print(f"[*] Retrieved & Reranked {len(docs)} highly relevant documents.")
    except Exception as e:
        print(f"[!] Retrieval failed after all retries: {e}")
        return "Error: Could not retrieve relevant documents due to database connectivity issues."
    
    # 2. Format Context
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    
    # 3. Generate Answer (with Fallbacks, Retry, and Rate Limiting automatically handled)
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt), 
            HumanMessage(content=question)
        ])
        
        print("\n[+] Answer:")
        print(response.content)
        return response.content
    except Exception as e:
        print(f"[!] Generation failed even after utilizing fallback models: {e}")
        return "Error: Could not generate an answer."

if __name__ == "__main__":
    test_question = "How did the introduction of the pregnancy checkbox on death certificates both improve and complicate the identification of maternal deaths, and what specific actions did CDC take between 2016 and 2020 to address the resulting data quality concerns before resuming publication of maternal mortality statistics?"
    answer_question_with_reranking(test_question)