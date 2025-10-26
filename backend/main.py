"""
Bible Verse Companion - FastAPI Backend
Step 1: Load ChromaDB + Embed user queries (using LangChain)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables FIRST (before any LangChain imports!)
load_dotenv()

# Set up LangSmith tracing (must happen before LangChain imports)
os.environ["LANGCHAIN_TRACING_V2"] = "true" if os.getenv("LANGCHAIN_API_KEY") else "false"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "bible-verse-companion"

# Now import LangChain modules
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tracers.context import tracing_v2_enabled
from langsmith import traceable

# Import reranker
from sentence_transformers import CrossEncoder

# Import for LLM
from openai import AsyncOpenAI

# FastAPI imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Configuration (hardcoded - no need for env vars)
CHROMA_DIR = "../data/output_bible_db_qwen"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
USE_RERANKER = True
QUERY_INSTRUCTION = "Represent the emotional or spiritual concern described by the user to retrieve comforting Bible passages:"
LLM_MODEL = "deepseek/deepseek-chat-v3.1:free"
LLM_TEMPERATURE = 0.7

# Retrieval settings
RETRIEVAL_K = 50  # Number of candidates from vector search
RETRIEVAL_N = 3   # Number of final verses to return (after reranking)

# Secrets (from .env)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Initialize OpenAI client for OpenRouter
openai_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# Print LangSmith status on startup
if os.getenv('LANGCHAIN_API_KEY'):
    print(f"✓ LangSmith tracing enabled - Project: bible-verse-companion")

# Global state for model and DB (loaded at startup)
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load embedding function and vector store at startup"""
    print("🚀 Starting up...")
    
    # Check ChromaDB path exists
    print(f"📁 Loading ChromaDB from: {CHROMA_DIR}")
    chroma_path = Path(CHROMA_DIR)
    if not chroma_path.exists():
        raise RuntimeError(f"ChromaDB not found at {CHROMA_DIR}")
    
    # Load embedding model (LangChain wrapper)
    print(f"🤖 Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={'device': 'cpu', 'trust_remote_code': True},
        encode_kwargs={'normalize_embeddings': True}
    )
    app_state["embeddings"] = embeddings
    print(f"   ✓ Embeddings loaded")
    
    # Load Chroma vector store
    print(f"📚 Loading Chroma vector store...")
    vectorstore = Chroma(
        persist_directory=str(chroma_path),
        embedding_function=embeddings,
        collection_name="bible_verses"
    )
    app_state["vectorstore"] = vectorstore
    
    # Get count
    collection_count = len(vectorstore.get()['ids'])
    print(f"   ✓ Loaded {collection_count} verse chunks")
    
    # Load reranker if enabled
    if USE_RERANKER:
        print(f"🎯 Loading reranker: {RERANK_MODEL}")
        reranker = CrossEncoder(RERANK_MODEL)
        app_state["reranker"] = reranker
        print(f"   ✓ Reranker loaded")
    
    print("✅ Ready to serve requests!\n")
    
    yield
    
    print("👋 Shutting down...")


app = FastAPI(
    title="Bible Verse Companion API",
    description="Semantic search for comforting Bible verses",
    version="0.1.0",
    lifespan=lifespan
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class RecommendRequest(BaseModel):
    issue: str


class Verse(BaseModel):
    ref: str
    text: str
    translation: str = "WEB"
    score: float


class RecommendResponse(BaseModel):
    verses: list[Verse]
    explanation: str = ""  # Empty for now, will add LLM later


# Health check
@app.get("/healthz")
async def health_check():
    """Health check endpoint"""
    vectorstore = app_state.get("vectorstore")
    db_chunks = 0
    if vectorstore:
        try:
            db_chunks = len(vectorstore.get()['ids'])
        except:
            pass
    
    return {
        "ok": True,
        "model": EMBED_MODEL,
        "reranker": RERANK_MODEL if USE_RERANKER else None,
        "reranker_enabled": USE_RERANKER,
        "db_chunks": db_chunks,
        "framework": "LangChain"
    }


# Helper functions with tracing
@traceable(run_type="retriever", name="search_bible_verses")
async def search_bible_verses(vectorstore, issue: str, k: int):
    """Search for relevant Bible verses using vector similarity"""
    query_with_instruction = f"{QUERY_INSTRUCTION} {issue}"
    
    results = vectorstore.similarity_search_with_score(
        query_with_instruction,
        k=k
    )
    
    return results


@traceable(run_type="tool", name="rerank_results")
def rerank_results(reranker, issue: str, results, n: int):
    """Rerank results using cross-encoder for better relevance"""
    # Prepare pairs for reranking (query, verse_text)
    pairs = [(issue, doc.page_content) for doc, _ in results]
    
    # Get reranker scores
    scores = reranker.predict(pairs)
    
    # Combine documents with new scores
    reranked = [(doc, float(score)) for (doc, _), score in zip(results, scores)]
    
    # Sort by reranker score (descending)
    reranked.sort(key=lambda x: x[1], reverse=True)
    
    # Return top n
    return reranked[:n]


@traceable(run_type="parser", name="format_verse_results")
def format_verse_results(results, use_reranker_scores: bool = False):
    """Format search results into verse objects"""
    verses = []
    for doc, score in results:
        if use_reranker_scores:
            # Reranker scores are already good (typically -10 to 10, higher is better)
            # Normalize to 0-1 range
            normalized_score = 1 / (1 + abs(score)) if score < 0 else min(score / 10, 1.0)
        else:
            # Convert distance to similarity score (Chroma uses L2 distance)
            normalized_score = 1 / (1 + score)
        
        verse = Verse(
            ref=doc.metadata.get('reference', 'Unknown'),
            text=doc.page_content,
            translation="WEB",
            score=normalized_score
        )
        verses.append(verse)
    
    return verses


@traceable(run_type="llm", name="generate_explanation")
async def generate_explanation(issue: str, verses: list[Verse]) -> str:
    """Generate empathetic explanation using LLM"""
    
    # Check for crisis keywords
    crisis_keywords = [
        "kill myself", "suicide", "end my life", "want to die", 
        "self-harm", "hurt myself", "cutting", "suicidal"
    ]
    if any(keyword in issue.lower() for keyword in crisis_keywords):
        return (
            "I'm deeply concerned about what you're going through. Please reach out for immediate support:\n\n"
            "• **National Suicide Prevention Lifeline**: 988 (24/7)\n"
            "• **Crisis Text Line**: Text HOME to 741741\n"
            "• **International Association for Suicide Prevention**: https://www.iasp.info/resources/Crisis_Centres/\n\n"
            "Your life has immeasurable value. Please don't face this alone—trained counselors are ready to help right now."
        )
    
    # Build prompt with verses
    verses_text = "\n\n".join([
        f"**{v.ref}** (World English Bible)\n\"{v.text[:300]}...\""  # Trim to ~300 chars
        for v in verses
    ])
    
    system_prompt = """You are a compassionate, non-denominational Christian guide helping people find comfort and encouragement through Scripture. Your responses should be:

- Warm, empathetic, and personal (use "you" language)
- Non-judgmental and supportive
- Focused on hope, comfort, and God's love
- 120-180 words in length
- Cite the verse references naturally in your response
- Avoid theological jargon or denominational teachings
- If the concern is serious but not crisis-level, be extra gentle

Your goal is to help the person feel seen, understood, and encouraged by connecting their concern to the timeless wisdom of Scripture."""

    user_prompt = f"""A person shared: "{issue}"

Here are the most relevant Bible passages:

{verses_text}

Write a compassionate 120-180 word response that:
1. Acknowledges their concern with empathy
2. Explains how these verses speak to their situation
3. Offers hope and encouragement
4. Naturally references the verses (e.g., "In Philippians 4:6-7, Paul reminds us...")

Remember: Be warm, personal, and focus on comfort rather than advice."""

    # Call OpenRouter API via OpenAI client
    try:
        completion = await openai_client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://bible-verse-companion.app",  # Update with your domain
                "X-Title": "Bible Verse Companion"
            },
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=300
        )
        
        explanation = completion.choices[0].message.content.strip()
        return explanation
            
    except Exception as e:
        print(f"Error calling LLM: {e}")
        # Fallback response
        verse_refs = ", ".join([v.ref for v in verses])
        return f"These verses ({verse_refs}) offer comfort for what you're experiencing. Take time to read and reflect on them—they contain timeless wisdom for your situation."


# Main recommend endpoint
@app.post("/recommend", response_model=RecommendResponse)
@traceable(run_type="chain", name="recommend_verses")
async def recommend_verses(request: RecommendRequest):
    """
    Get Bible verse recommendations based on user's concern
    
    Step 1: Embed query + retrieve from ChromaDB using LangChain (no reranker yet, no LLM yet)
    """
    
    # Validate input
    if not request.issue or not request.issue.strip():
        raise HTTPException(status_code=400, detail="Issue cannot be empty")
    
    vectorstore = app_state.get("vectorstore")
    reranker = app_state.get("reranker")
    
    if not vectorstore:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    # Step 1: Retrieve top K candidates from vector store
    results = await search_bible_verses(vectorstore, request.issue, RETRIEVAL_K)
    
    # Check if we found anything
    if not results:
        raise HTTPException(status_code=404, detail="No verses found")
    
    # Step 2: Rerank if enabled
    if USE_RERANKER and reranker:
        reranked_results = rerank_results(reranker, request.issue, results, RETRIEVAL_N)
        verses = format_verse_results(reranked_results, use_reranker_scores=True)
    else:
        # No reranking, just take top n from vector search
        verses = format_verse_results(results[:RETRIEVAL_N], use_reranker_scores=False)
    
    # Step 3: Generate empathetic explanation with LLM
    explanation = await generate_explanation(request.issue, verses)
    
    return RecommendResponse(
        verses=verses,
        explanation=explanation
    )


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Bible Verse Companion API",
        "version": "0.2.0",
        "status": "Step 2: Reranker added ✅",
        "framework": "LangChain",
        "reranker_enabled": USE_RERANKER,
        "endpoints": {
            "health": "/healthz",
            "recommend": "POST /recommend",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = 8080
    uvicorn.run(app, host="0.0.0.0", port=port)

