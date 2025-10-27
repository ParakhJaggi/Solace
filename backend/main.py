"""
Bible Verse Companion - FastAPI Backend (Pinecone Version)
Ultra-lightweight: ~200MB memory (no local embeddings!)
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangSmith setup (optional)
os.environ["LANGCHAIN_TRACING_V2"] = "true" if os.getenv("LANGCHAIN_API_KEY") else "false"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "bible-verse-companion"

from langsmith import traceable
from pinecone import Pinecone
from openai import AsyncOpenAI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_HOST = "solace-t42ww4d.svc.aped-4627-b74a.pinecone.io"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_MODEL = "deepseek/deepseek-chat-v3.1:free"
LLM_TEMPERATURE = 0.7

# Retrieval settings
RETRIEVAL_K = 50  # Get top 50 candidates from Pinecone for better quality
RETRIEVAL_N = 3   # Return top 3 to user

# Initialize clients
openai_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# Print startup info
if os.getenv('LANGCHAIN_API_KEY'):
    print(f"✓ LangSmith tracing enabled")

# Global state
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Pinecone client at startup"""
    print("🚀 Starting up...")
    
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not set")
    
    # Initialize Pinecone
    print(f"📊 Connecting to Pinecone index: {PINECONE_INDEX_HOST}")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(host=PINECONE_INDEX_HOST)
    
    # Store in app state
    app_state["pinecone_index"] = index
    
    # Get stats
    stats = index.describe_index_stats()
    total_vectors = stats.get('total_vector_count', 0)
    print(f"   ✓ Connected! {total_vectors:,} verses indexed")
    
    print("✅ Ready to serve requests!\n")
    
    yield
    
    print("👋 Shutting down...")


app = FastAPI(
    title="Bible Verse Companion API",
    description="Semantic search for comforting Bible verses",
    version="0.3.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    explanation: str


# Health check
@app.get("/healthz")
async def health_check():
    """Health check endpoint"""
    index = app_state.get("pinecone_index")
    db_count = 0
    if index:
        try:
            stats = index.describe_index_stats()
            db_count = stats.get('total_vector_count', 0)
        except:
            pass
    
    return {
        "ok": True,
        "db_verses": db_count,
        "framework": "Pinecone + DeepSeek"
    }


# Helper functions with tracing
@traceable(run_type="retriever", name="search_pinecone")
async def search_pinecone(index, query: str, k: int):
    """Search Pinecone for relevant verses (integrated embedding)"""
    # Pinecone automatically embeds the query text!
    results = index.search(
        namespace="__default__",
        query={
            "inputs": {"text": query},
            "top_k": k
        },
        fields=["text", "reference", "book", "book_name", "testament", "translation"]
    )
    
    # It's a Pydantic object - access attributes directly
    matches = []
    if hasattr(results, 'result'):
        result_obj = results.result
        if result_obj and hasattr(result_obj, 'hits'):
            matches = result_obj.hits or []
    elif hasattr(results, 'matches'):
        matches = results.matches or []
    
    return matches


@traceable(run_type="parser", name="format_results")
def format_results(matches, n: int):
    """Format Pinecone results into verse objects"""
    verses = []
    
    for match in matches[:n]:
        # Access Pydantic object attributes
        score = getattr(match, '_score', getattr(match, 'score', 0))
        fields = getattr(match, 'fields', {})
        
        # Fields might be a dict or Pydantic object
        if hasattr(fields, '__dict__'):
            fields_dict = fields.__dict__
        elif isinstance(fields, dict):
            fields_dict = fields
        else:
            fields_dict = {}
        
        verse = Verse(
            ref=fields_dict.get('reference', 'Unknown'),
            text=fields_dict.get('text', ''),
            translation=fields_dict.get('translation', 'WEB'),
            score=float(score) if score else 0.0
        )
        verses.append(verse)
    
    return verses


def clean_llm_output(text: str) -> str:
    """Remove special tokens and artifacts from LLM output"""
    import re
    
    # Remove common special tokens
    patterns = [
        r'<\|.*?\|>',  # <|begin_of_sentence|>, <|end|>, etc.
        r'<｜.*?｜>',  # Wide-character variants
        r'\[INST\].*?\[/INST\]',  # Instruction tokens
        r'<s>|</s>',  # Sentence markers
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL)
    
    # Remove common acknowledgment phrases at the start
    acknowledgments = [
        r'^Of course\.\s*',
        r'^Sure\.\s*',
        r'^Certainly\.\s*',
        r'^Here is (?:a|the) response.*?[:\.]?\s*\n*',
        r'^Here\'s (?:a|the) response.*?[:\.]?\s*\n*',
        r'^(?:Here is|Here\'s) (?:a|an) .*?[:\.]?\s*\n*',
        r'^Based on (?:the|your).*?[:\.]?\s*\n*',
    ]
    
    for ack in acknowledgments:
        text = re.sub(ack, '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove personal closings at the end
    personal_closings = [
        r'\s*I\'ll be (?:thinking of|praying for) you\.?\s*$',
        r'\s*I\'m thinking of you\.?\s*$',
        r'\s*(?:Thinking|Praying) of you\.?\s*$',
        r'\s*You\'re in my (?:thoughts|prayers)\.?\s*$',
        r'\s*Blessings to you\.?\s*$',
        r'\s*May (?:God|the Lord) bless you\.?\s*$',
    ]
    
    for closing in personal_closings:
        text = re.sub(closing, '', text, flags=re.IGNORECASE)
    
    # Remove word count annotations at the end
    text = re.sub(r'\n*\*?\(?\*?Word Count:?\s*\d+\*?\)?\*?\s*$', '', text, flags=re.IGNORECASE)
    
    return text.strip()


@traceable(run_type="llm", name="generate_explanation")
async def generate_explanation(issue: str, verses: list[Verse]) -> str:
    """Generate empathetic explanation using LLM"""
    
    # Crisis detection
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
    
    # Build prompt
    verses_text = "\n\n".join([
        f"**{v.ref}** (World English Bible)\n\"{v.text[:300]}...\""
        for v in verses
    ])
    
    system_prompt = """You are a compassionate, non-denominational Christian guide. Write 2-4 paragraphs of comfort and encouragement using Scripture.

Guidelines:
- Start immediately with empathy and understanding
- Weave in the Bible verses naturally (use **bold** for verse references)
- Focus on hope, comfort, and God's love
- Use warm, personal "you" language
- Avoid: theological jargon, personal sign-offs, meta-commentary
- Do NOT say things like "I'm thinking of you", "I'll be praying for you", or similar personal closings

Just write the encouragement directly."""

    user_prompt = f"""Person's concern: "{issue}"

Relevant Scripture:

{verses_text}

Write your response (2-4 paragraphs)."""

    # Call LLM
    try:
        completion = await openai_client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://solace.app",
                "X-Title": "Solace"
            },
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=600
        )
        
        explanation = completion.choices[0].message.content.strip()
        # Clean special tokens and artifacts
        explanation = clean_llm_output(explanation)
        return explanation
            
    except Exception as e:
        print(f"Error calling LLM: {e}")
        verse_refs = ", ".join([v.ref for v in verses])
        return f"These verses ({verse_refs}) offer comfort for what you're experiencing. Take time to read and reflect on them—they contain timeless wisdom for your situation."


# Main endpoint
@app.post("/recommend", response_model=RecommendResponse)
@traceable(run_type="chain", name="recommend_verses")
async def recommend_verses(request: RecommendRequest):
    """
    Get Bible verse recommendations based on user's concern
    
    Uses Pinecone for serverless vector search (no local embeddings!)
    """
    
    # Validate input
    if not request.issue or not request.issue.strip():
        raise HTTPException(status_code=400, detail="Issue cannot be empty")
    
    if len(request.issue) > 500:
        raise HTTPException(status_code=400, detail="Input too long. Please keep your message under 500 characters.")
    
    index = app_state.get("pinecone_index")
    
    if not index:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    # Step 1: Search Pinecone (it handles embedding automatically)
    matches = await search_pinecone(index, request.issue, RETRIEVAL_K)
    
    if not matches:
        raise HTTPException(status_code=404, detail="No verses found")
    
    # Step 2: Format top N results
    verses = format_results(matches, RETRIEVAL_N)
    
    # Step 3: Generate explanation with LLM
    explanation = await generate_explanation(request.issue, verses)
    
    return RecommendResponse(
        verses=verses,
        explanation=explanation
    )


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Solace - Bible Verse Companion API",
        "version": "0.3.0",
        "status": "Pinecone + DeepSeek ✅",
        "memory": "~200MB (serverless embeddings)",
        "endpoints": {
            "health": "/healthz",
            "recommend": "POST /recommend",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)

