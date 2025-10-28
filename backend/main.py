"""
Solace - FastAPI Backend

Find comfort in the texts you love.
Supports: Bible (Christian/Jewish) and Harry Potter

Ultra-lightweight: ~200MB memory (serverless embeddings via Pinecone)
"""

import os
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangSmith setup (optional)
os.environ["LANGCHAIN_TRACING_V2"] = "true" if os.getenv("LANGCHAIN_API_KEY") else "false"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "solace"

from langsmith import traceable
from tavily import TavilyClient
from pinecone import Pinecone
from openai import AsyncOpenAI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_HOST = "solace-t42ww4d.svc.aped-4627-b74a.pinecone.io"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
LLM_MODEL = "meta-llama/llama-3.3-70b-instruct:free"  # Good balance of speed & quality
LLM_TEMPERATURE = 0.7

# Retrieval settings
RETRIEVAL_K = 50  # Get top 50 candidates from Pinecone for better quality
RETRIEVAL_N = 3   # Return top 3 to user
USE_RERANKER = True  # Use Pinecone's hosted reranker 

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
    
    # Initialize Tavily search tool
    if TAVILY_API_KEY:
        tavily_client = TavilyClient(TAVILY_API_KEY)
        app_state["tavily_client"] = tavily_client
        print(f"   ✓ Tavily client initialized")
    else:
        print(f"   ⚠️  Tavily API key not set - social media search disabled")
    
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
    tradition: str = "christian"  # "christian" or "jewish"


class Verse(BaseModel):
    ref: str
    text: str
    translation: str = "WEB"
    score: float
    book_name: str = ""  # For diversity filtering
    url: str = ""  # For social media links
    
    class Config:
        # Ensure all fields are included in serialization
        exclude_unset = False


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
        "framework": "Pinecone + DeepSeek",
        "reranker": "pinecone-rerank-v0" if USE_RERANKER else "none"
    }


# Helper functions
@traceable(run_type="tool", name="search_twitter")
async def search_twitter_content(query: str, tavily_client):
    """Search Twitter for relevant comfort and encouragement posts"""
    try:
        # Use the exact settings that work
        response = tavily_client.search(
            query=query,
            include_raw_content="text",
            include_domains=["x.com"]
        )
        
        # Process results into tweet-like format
        tweets = []
        if response and 'results' in response:
            for result in response['results']:
                # Extract tweet content and metadata
                content = result.get('content', '') or result.get('raw_content', '')
                url = result.get('url', '')
                title = result.get('title', '')
                
                # Only process X.com URLs (since we're only searching x.com)
                if not ('x.com/' in url):
                    continue
                
                # Skip if content is just the JavaScript disabled message
                if "JavaScript is disabled" in content or "enable JavaScript" in content:
                    continue
                
                # Skip if content is too short or doesn't look like a tweet
                if len(content.strip()) < 20:
                    continue
                
                
                # Extract username from X.com URL
                username = "Unknown"
                if 'x.com/' in url:
                    try:
                        username = url.split('x.com/')[1].split('/')[0]
                        if username.startswith('@'):
                            username = username[1:]  # Remove @ if present
                    except:
                        pass
                
                tweets.append({
                    'content': content,
                    'username': username,
                    'url': url,
                    'platform': 'Twitter'
                })
        
        return tweets
        
    except Exception as e:
        print(f"Error searching Twitter: {e}")
        return []


@traceable(run_type="retriever", name="search_pinecone")
async def search_pinecone(index, query: str, k: int, testament_filter: list = None):
    """Search Pinecone for relevant verses (integrated embedding)"""
    # Pinecone automatically embeds the query text!
    search_params = {
        "namespace": "__default__",
        "query": {
            "inputs": {"text": query},
            "top_k": k
        },
        "fields": ["text", "reference", "book", "book_name", "testament", "translation"]
    }
    
    # Add testament filter if specified (as part of query for integrated embedding)
    if testament_filter:
        search_params["query"]["filter"] = {"testament": {"$in": testament_filter}}
    
    results = index.search(**search_params)
    
    # It's a Pydantic object - access attributes directly
    matches = []
    if hasattr(results, 'result'):
        result_obj = results.result
        if result_obj and hasattr(result_obj, 'hits'):
            matches = result_obj.hits or []
    elif hasattr(results, 'matches'):
        matches = results.matches or []
    
    return matches


@traceable(run_type="tool", name="rerank_results")
def rerank_results(query: str, matches, top_n: int):
    """Rerank results using Pinecone's hosted reranker"""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Convert matches to documents for reranking
    documents = []
    for match in matches:
        fields = getattr(match, 'fields', {})
        if hasattr(fields, '__dict__'):
            fields_dict = fields.__dict__
        elif isinstance(fields, dict):
            fields_dict = fields
        else:
            fields_dict = {}
        
        # Extract id from match
        match_id = getattr(match, '_id', getattr(match, 'id', 'unknown'))
        
        documents.append({
            "id": match_id,
            "text": fields_dict.get('text', ''),
            "reference": fields_dict.get('reference', ''),
            "book_name": fields_dict.get('book_name', ''),
            "translation": fields_dict.get('translation', 'WEB')
        })
    
    # Call Pinecone's hosted reranker
    reranked = pc.inference.rerank(
        model="pinecone-rerank-v0",
        query=query,
        documents=documents,
        top_n=top_n,
        rank_fields=["text"],
        return_documents=True,
        parameters={"truncate": "END"}
    )
    
    # Convert reranked results back to match-like objects
    reranked_matches = []
    for item in reranked.data:
        # Create a simple object with the reranked data
        class RankedMatch:
            def __init__(self, doc, score):
                self._id = doc.get('id', 'unknown')
                self._score = score
                self.fields = {
                    'text': doc.get('text', ''),
                    'reference': doc.get('reference', ''),
                    'book_name': doc.get('book_name', ''),
                    'translation': doc.get('translation', 'WEB')
                }
        
        reranked_matches.append(RankedMatch(item.document, item.score))
    
    return reranked_matches


@traceable(run_type="parser", name="format_results")
def format_results(matches, n: int, ensure_diversity: bool = True):
    """Format Pinecone results into verse objects with optional book diversity"""
    all_verses = []
    
    # First, convert all matches to verse objects
    for match in matches:
        score = getattr(match, '_score', getattr(match, 'score', 0))
        fields = getattr(match, 'fields', {})
        
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
        # Store book name for diversity check
        verse.book_name = fields_dict.get('book_name', '')
        all_verses.append(verse)
    
    if not ensure_diversity or len(all_verses) <= n:
        return all_verses[:n]
    
    # Apply diversity: prefer different books
    selected_verses = []
    seen_books = set()
    
    # First pass: one verse per book (up to n)
    for verse in all_verses:
        if len(selected_verses) >= n:
            break
        if verse.book_name not in seen_books:
            selected_verses.append(verse)
            seen_books.add(verse.book_name)
    
    # Second pass: fill remaining slots with highest scores (can repeat books)
    if len(selected_verses) < n:
        for verse in all_verses:
            if len(selected_verses) >= n:
                break
            if verse not in selected_verses:
                selected_verses.append(verse)
    
    return selected_verses[:n]


def clean_llm_output(text: str) -> str:
    """Remove special tokens and artifacts from LLM output"""
    
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


@app.post("/recommend/stream")
async def recommend_verses_stream(request: RecommendRequest):
    """
    Stream passage recommendations with real-time LLM generation
    
    Returns Server-Sent Events (SSE) stream with verses + streaming explanation
    """
    import json
    
    # Validate input
    if not request.issue or not request.issue.strip():
        raise HTTPException(status_code=400, detail="Issue cannot be empty")
    
    if len(request.issue) > 500:
        raise HTTPException(status_code=400, detail="Input too long. Please keep your message under 500 characters.")
    
    # Crisis detection - check for self-harm or suicide language
    crisis_keywords = [
        "kill myself", "suicide", "end my life", "want to die", 
        "self-harm", "hurt myself", "cutting", "suicidal",
        "end it all", "better off dead", "no reason to live"
    ]
    is_crisis = any(keyword in request.issue.lower() for keyword in crisis_keywords)
    
    if is_crisis:
        # Return crisis resources immediately, skip all verse search
        async def crisis_stream():
            crisis_message = (
                "I'm deeply concerned about what you're going through. **Please reach out for immediate support:**\n\n"
                "• **National Suicide Prevention Lifeline**: **988** (24/7, call or text)\n"
                "• **Crisis Text Line**: Text **HOME** to **741741**\n"
                "• **International Association for Suicide Prevention**: https://www.iasp.info/resources/Crisis_Centres/\n\n"
                "Your life has immeasurable value. Please don't face this alone—trained counselors are ready to help right now. "
                "They understand what you're going through and can provide the support you need."
            )
            
            # Send as explanation chunks
            yield f"data: {json.dumps({'type': 'crisis', 'content': crisis_message})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
        return StreamingResponse(crisis_stream(), media_type="text/event-stream")
    
    index = app_state.get("pinecone_index")
    
    if not index:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    # Handle social media flow
    if request.tradition == "social_media":
        tavily_client = app_state.get("tavily_client")
        if not tavily_client:
            raise HTTPException(status_code=503, detail="Social media search not available")
        
        # Wrap the social media process in a traceable context
        @traceable(run_type="chain", name="recommend_social_media_stream")
        async def execute_social_media_recommendation():
            # Step 1: Search Twitter
            tweets = await search_twitter_content(request.issue, tavily_client)
            
            if not tweets:
                return None, "No relevant social media content found"
            
            # Format tweets as verses
            verses = []
            for i, tweet in enumerate(tweets):
                verses.append(Verse(
                    ref=f"@{tweet['username']}",
                    text=tweet['content'][:1500] + "..." if len(tweet['content']) > 1500 else tweet['content'],
                    translation=tweet['platform'],
                    score=0.9 - (i * 0.1),  # Decreasing relevance score
                    book_name=tweet['username'],
                    url=tweet['url']
                ))
            
            return verses, None
        
        async def generate_social_stream():
            try:
                # Execute social media retrieval
                verses, error = await execute_social_media_recommendation()
                
                if error:
                    yield f"data: {json.dumps({'error': error})}\n\n"
                    return
                
                # Send verses first
                verses_data = {
                    "type": "verses",
                    "verses": [
                        {
                            "ref": v.ref,
                            "text": v.text,
                            "translation": v.translation,
                            "score": v.score,
                            "url": v.url
                        } for v in verses
                    ]
                }
                yield f"data: {json.dumps(verses_data)}\n\n"
                
                # Step 2: Stream LLM explanation for social media
                @traceable(run_type="llm", name="generate_social_explanation_stream")
                async def create_social_llm_stream():
                    # Build prompt for social media content
                    tweets_text = "\n\n".join([
                        f"@{v.ref}: \"{v.text}\""
                        for v in verses
                    ])
                    tweet_refs = ", ".join([v.ref for v in verses])
                    
                    system_prompt = """You are a compassionate guide who finds wisdom in social media. Write 2-4 paragraphs of comfort and encouragement using ONLY the Twitter posts provided.

CRITICAL RULES:
- ONLY reference the specific tweets provided below - DO NOT mention any other content
- Use **bold** for the exact usernames INLINE within sentences (e.g., "as @username shared")
- NEVER put usernames on their own separate line
- Start immediately with empathy and understanding
- Focus on how these real people's experiences can help and encourage
- Use warm, personal "you" language
- Reference the authenticity and relatability of social media wisdom
- Avoid: made-up content, personal sign-offs
- Do NOT say "I'm thinking of you", "I'll be praying for you"
- FORMAT: Write in 2-4 separate paragraphs with blank lines between them

Just write the encouragement directly using ONLY the provided tweets."""
                    
                    user_prompt = f"""Person's concern: "{request.issue}"

The ONLY tweets you may reference are: {tweet_refs}

Here are the tweets:

{tweets_text}

Write your response (2-4 paragraphs) using ONLY these tweets."""
                    
                    # Stream from LLM
                    return await openai_client.chat.completions.create(
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
                        max_tokens=600,
                        stream=True
                    )
                
                # Get the stream within traced context
                stream = await create_social_llm_stream()
                
                # Accumulate chunks for cleaning
                full_text = ""
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_text += content
                        
                        # Clean content in real-time to remove special tokens
                        cleaned_content = content
                        # Remove special tokens from chunks
                        special_tokens = [
                            r'<\|begin_of_sentence\|>',
                            r'<｜begin▁of▁sentence｜>',
                            r'<｜begin of sentence｜>',
                            r'<\|end_of_text\|>',
                            r'<｜end▁of▁text｜>',
                            r'<eos>',
                            r'</s>'
                        ]
                        for token in special_tokens:
                            cleaned_content = re.sub(token, '', cleaned_content, flags=re.IGNORECASE)
                        
                        # Only send if there's content after cleaning
                        if cleaned_content:
                            data = {
                                "type": "explanation_chunk",
                                "content": cleaned_content
                            }
                            yield f"data: {json.dumps(data)}\n\n"
                
                # Send done signal
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                
            except Exception as e:
                print(f"Error in social media stream: {e}", flush=True)
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(generate_social_stream(), media_type="text/event-stream")
    
    # Determine testament filter for traditional sources
    testament_filter = None
    if request.tradition == "jewish":
        testament_filter = ["OT"]
    elif request.tradition == "christian":
        testament_filter = ["OT", "NT"]
    elif request.tradition == "harry_potter":
        testament_filter = ["HP"]
    
    # Wrap the entire streaming process in a traceable context
    @traceable(run_type="chain", name="recommend_verses_stream")
    async def execute_recommendation():
        # Step 1: Search Pinecone
        matches = await search_pinecone(index, request.issue, RETRIEVAL_K, testament_filter)
        
        if not matches:
            return None, "No verses found"
        
        # Step 2: Rerank
        if USE_RERANKER:
            try:
                matches = rerank_results(request.issue, matches, RETRIEVAL_N)
                verses = format_results(matches, RETRIEVAL_N, ensure_diversity=False)
            except Exception as e:
                print(f"⚠️  Reranker failed ({e}), falling back to diversity filter", flush=True)
                verses = format_results(matches, RETRIEVAL_N, ensure_diversity=True)
        else:
            verses = format_results(matches, RETRIEVAL_N, ensure_diversity=True)
        
        return verses, None
    
    async def generate_stream():
        try:
            # Execute all retrieval steps within tracing context
            verses, error = await execute_recommendation()
            
            if error:
                yield f"data: {json.dumps({'error': error})}\n\n"
                return
            
            # Send verses first
            verses_data = {
                "type": "verses",
                "verses": [
                    {
                        "ref": v.ref,
                        "text": v.text,
                        "translation": v.translation,
                        "score": v.score,
                        "url": v.url
                    } for v in verses
                ]
            }
            yield f"data: {json.dumps(verses_data)}\n\n"
            
            # Step 3: Stream LLM explanation (wrapped in traceable context)
            @traceable(run_type="llm", name="generate_explanation_stream")
            async def create_llm_stream():
                # Build prompt - format verses WITHOUT bold so LLM doesn't mimic that pattern
                verses_text = "\n\n".join([
                    f"{v.ref}{' (' + v.translation + ')' if v.translation != 'Original' else ''}: \"{v.text[:300]}...\""
                    for v in verses
                ])
                verse_refs = ", ".join([v.ref for v in verses])
                
                # Get system prompt based on tradition
                if request.tradition == "jewish":
                    system_prompt = """You are a compassionate Jewish guide. Write 2-4 paragraphs of comfort and encouragement using ONLY the Torah/Tanakh verses provided.

CRITICAL RULES:
- ONLY reference the specific verses provided below - DO NOT mention any other verses
- Use **bold** for the exact verse references INLINE within sentences (e.g., "as it says in **Psalm 55:22**")
- NEVER put verse references on their own separate line
- Start immediately with empathy and understanding
- Focus on hope, comfort, and Hashem's love
- Use warm, personal "you" language
- Reference Jewish concepts naturally (Torah, mitzvot, tikkun olam)
- Avoid: Christian terminology, New Testament, made-up citations
- Do NOT say "I'm thinking of you", "I'll be praying for you"
- FORMAT: Write in 2-4 separate paragraphs with blank lines between them

Just write the encouragement directly using ONLY the provided verses."""
                elif request.tradition == "harry_potter":
                    system_prompt = """You are a compassionate guide who finds wisdom in stories. Write 2-4 paragraphs of comfort and encouragement using ONLY the Harry Potter passages provided.

CRITICAL RULES:
- ONLY reference the specific passages provided below - DO NOT mention any other scenes
- Use **bold** for the exact references INLINE within sentences (e.g., "as we see in **Deathly Hallows, Chapter 33**")
- NEVER put references on their own separate line
- Start immediately with empathy and understanding
- Draw parallels between their situation and the themes in the passages (courage, friendship, overcoming fear, belonging, loss, hope)
- Reference characters, moments, and themes naturally (Harry's courage, Dumbledore's wisdom, the power of friendship)
- Use warm, personal "you" language - connect the story's wisdom to their life
- Avoid: religious language, made-up scenes, personal sign-offs
- Do NOT say "I'm thinking of you", "May you find peace"
- FORMAT: Write in 2-4 separate paragraphs with blank lines between them

Just write the encouragement directly, connecting the story's wisdom to their experience using ONLY the provided passages."""
                else:  # christian
                    system_prompt = """You are a compassionate, non-denominational Christian guide. Write 2-4 paragraphs of comfort and encouragement using ONLY the Bible verses provided.

CRITICAL RULES:
- ONLY reference the specific verses provided below - DO NOT mention any other verses
- Use **bold** for the exact verse references INLINE within sentences (e.g., "as it says in **Psalm 23:1**")
- NEVER put verse references on their own separate line
- Start immediately with empathy and understanding
- Focus on hope, comfort, and God's love
- Use warm, personal "you" language
- Avoid: theological jargon, made-up citations, personal sign-offs
- Do NOT say "I'm thinking of you", "I'll be praying for you"
- FORMAT: Write in 2-4 separate paragraphs with blank lines between them

Just write the encouragement directly using ONLY the provided verses."""
                
                user_prompt = f"""Person's concern: "{request.issue}"

The ONLY passages you may reference are: {verse_refs}

Here are the passages:

{verses_text}

Write your response (2-4 paragraphs) using ONLY these passages."""
                
                # Stream from LLM
                return await openai_client.chat.completions.create(
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
                    max_tokens=600,
                    stream=True
                )
            
            # Get the stream within traced context
            stream = await create_llm_stream()
            
            # Accumulate chunks for cleaning
            full_text = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_text += content
                    
                    # Clean content in real-time to remove special tokens
                    cleaned_content = content
                    # Remove special tokens from chunks
                    special_tokens = [
                        r'<\|begin_of_sentence\|>',
                        r'<｜begin▁of▁sentence｜>',
                        r'<｜begin of sentence｜>',
                        r'<\|end_of_text\|>',
                        r'<｜end▁of▁text｜>',
                        r'<eos>',
                        r'</s>'
                    ]
                    for token in special_tokens:
                        cleaned_content = re.sub(token, '', cleaned_content, flags=re.IGNORECASE)
                    
                    # Only send if there's content after cleaning
                    if cleaned_content:
                        data = {
                            "type": "explanation_chunk",
                            "content": cleaned_content
                        }
                        yield f"data: {json.dumps(data)}\n\n"
            
            # Send done signal
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            print(f"Error in stream: {e}", flush=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate_stream(), media_type="text/event-stream")


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Solace - Find Comfort in the Texts You Love",
        "version": "0.6.0",
        "status": "Pinecone + DeepSeek V3.1 + Streaming ✅",
        "sources": ["Bible (Christian)", "Torah/Tanakh (Jewish)", "Harry Potter", "Social Media (Twitter)"],
        "memory": "~200MB (serverless embeddings)",
        "features": ["Streaming responses", "Real-time LLM generation", "Book diversity filtering"],
        "endpoints": {
            "health": "/healthz",
            "recommend": "POST /recommend/stream (streaming SSE)",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)

