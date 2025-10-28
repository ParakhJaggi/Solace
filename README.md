# Solace

**Find comfort in the texts you love.**

A RAG (Retrieval Augmented Generation) application that helps users find comforting passages when they're going through difficult times. Search across religious texts (Bible) or beloved stories (Harry Potter) and receive personalized, empathetic explanations.

## Core Value Proposition

**User types what they're going through → gets 3 highly relevant passages + 2-4 paragraphs of empathetic explanation.**

### Example Use Cases

1. **"I'm anxious about work."** → Relevant passages + warm encouragement
2. **"I feel like an outsider."** → Passages about belonging + comforting context
3. **"I'm grieving a loss."** → Passages about hope and healing + empathetic support

## Features

### Multi-Source Retrieval
- **Christian Bible** (Old & New Testament) - 31,000+ verses
- **Jewish Texts** (Torah/Tanakh only) - Old Testament filtering
- **Harry Potter** - 7 books, ~6,000+ passages
- **Social Media** - Real tweets and posts for comfort

### Intelligent Search Pipeline
- **Semantic search** using Pinecone's `nvidia/llama-text-embed-v2` (1024-dim embeddings)
- **Two-stage retrieval**: Vector search (k=50) → Reranking (n=3) with `pinecone-rerank-v0`
- **Book diversity filter**: Prevents all results from same book (e.g., all Psalms)
- **Metadata filtering**: Testament-based filtering (OT, NT, HP)

### AI-Powered Synthesis
- **LLM explanations** using DeepSeek V3.1 (2-4 paragraphs, ~200-300 words)
- **Tradition-aware prompts**: Different tone for Jewish/Christian/Harry Potter contexts
- **Crisis detection**: Detects self-harm language and provides hotline resources
- **Moderation handling**: Graceful fallback for false-positive content flags

### Production-Ready
- **FastAPI backend** with LangSmith tracing
- **Next.js frontend** with static site generation
- **Character limits** (500 chars) with validation
- **Error handling**: Graceful fallbacks for rate limits, moderation, etc.
- **Enter key submission** for better UX
- **Deployed** on Render (frontend + backend)

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- Pinecone account (free tier)
- OpenRouter account (free tier)
- Tavily account (for social media search)

### 1. Set Up the Vector Database

```bash
cd data

# Download Harry Potter CSV
curl -o books/harry_potter_books.csv \
  "https://raw.githubusercontent.com/gastonstat/harry-potter-data/refs/heads/main/csv-data-file/harry_potter_books.csv"

# Create .env file
echo "PINECONE_API_KEY=your_key_here" > .env

# Install dependencies
pip install -r requirements_pinecone.txt

# Embed Bible verses
python embed_bible_pinecone.py

# Embed Harry Potter passages
python embed_harry_potter_pinecone.py
```

Expected output:
- **Bible**: ~31,000 verse chunks → Pinecone (testament: OT/NT)
- **Harry Potter**: ~6,000 passages → Pinecone (testament: HP)
- Takes ~30-60 minutes with rate limiting

### 2. Run the Backend

```bash
cd backend

# Create .env file
cat > .env << EOF
PINECONE_API_KEY=your_pinecone_key
OPENROUTER_API_KEY=your_openrouter_key
TAVILY_API_KEY=your_tavily_key
LANGCHAIN_API_KEY=your_langsmith_key  # Optional for tracing
EOF

# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py
```

Backend runs on `http://localhost:8080`

### 3. Run the Frontend

```bash
cd frontend

# Install dependencies
npm install

# Create .env.local file
echo "NEXT_PUBLIC_API_URL=http://localhost:8080" > .env.local

# Run dev server
npm run dev
```

Frontend runs on `http://localhost:3000`

## Architecture

### Data Flow

```
User Input: "I'm anxious about work"
    │
    ▼
┌─────────────────────────────────────────┐
│ FastAPI Backend                         │
│                                         │
│ 1. Input validation (500 char limit)   │
│ 2. Testament filter selection           │
│    • jewish → ["OT"]                    │
│    • christian → ["OT", "NT"]           │
│    • harry_potter → ["HP"]              │
│    • social_media → Tavily search       │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│ Pinecone Search (Integrated Embedding)  │
│                                         │
│ • Model: nvidia/llama-text-embed-v2 (1024-dim) │
│ • Query embedded automatically          │
│ • Metadata filter applied               │
│ • Returns top k=50 candidates           │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│ Pinecone Reranker (pinecone-rerank-v0)  │
│                                         │
│ • Re-scores 50 candidates               │
│ • Returns top n=3 most relevant         │
│ • Graceful fallback if quota exceeded   │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│ Book Diversity Filter (Optional)        │
│                                         │
│ • If reranker fails, ensures variety    │
│ • Picks one passage per book            │
│ • Prevents "all Psalms" results         │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│ LLM Synthesis (DeepSeek V3.1 via        │
│ OpenRouter)                             │
│                                         │
│ • Tradition-aware prompts:              │
│   - Jewish: Torah/Tanakh language       │
│   - Christian: Non-denominational       │
│   - Harry Potter: Story wisdom          │
│   - Social Media: Real experiences      │
│ • Crisis detection + hotlines           │
│ • Moderation retry logic                │
│ • 2-4 paragraphs of comfort             │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│ Response                                │
│                                         │
│ {                                       │
│   "verses": [                           │
│     {                                   │
│       "ref": "Philippians 4:6-7",       │
│       "text": "...",                    │
│       "translation": "WEB",             │
│       "score": 0.89,                    │
│       "url": "https://..."              │
│     }                                   │
│   ],                                    │
│   "explanation": "..."                  │
│ }                                       │
└─────────────────────────────────────────┘
```

## Technology Stack

### Backend
- **Framework**: FastAPI
- **Embeddings**: Pinecone Inference API (`nvidia/llama-text-embed-v2`, 1024 dimensions)
- **Vector Database**: Pinecone (serverless)
- **Reranker**: Pinecone Rerank (`pinecone-rerank-v0`)
- **LLM**: DeepSeek V3.1 via OpenRouter
- **Social Search**: Tavily API
- **Tracing**: LangSmith (optional)
- **Deployment**: Render

### Frontend
- **Framework**: Next.js 14 (React)
- **Styling**: Tailwind CSS
- **Build**: Static Site Generation (SSG)
- **Deployment**: Render Static Site

### Data
- **Bible**: World English Bible (WEB) - Public Domain XML
- **Harry Potter**: [gastonstat/harry-potter-data](https://github.com/gastonstat/harry-potter-data) CSV
- **Social Media**: Real-time Twitter/X content via Tavily
- **Chunking**: 3 verses (Bible) / 10 lines (Harry Potter)

## Project Structure

```
ask-book/
├── data/
│   ├── books/
│   │   └── harry_potter_books.csv      # HP source data (95K lines)
│   ├── engwebp_vpl.xml                 # Bible XML source (31K verses)
│   ├── embed_bible_pinecone.py         # Bible → Pinecone embedder
│   ├── embed_harry_potter_pinecone.py  # Harry Potter → Pinecone embedder
│   ├── requirements_pinecone.txt       # Embedding dependencies
│   └── .env                            # PINECONE_API_KEY
│
├── backend/
│   ├── main.py                         # FastAPI app
│   ├── requirements.txt                # Backend dependencies
│   └── .env                            # API keys
│
├── frontend/
│   ├── app/
│   │   ├── page.js                     # Main app page
│   │   ├── layout.js                   # Root layout
│   │   └── globals.css                 # Global styles
│   ├── tailwind.config.js              # Tailwind config
│   ├── next.config.js                  # Next.js config (SSG)
│   ├── package.json                    # Frontend dependencies
│   └── .env.local                      # NEXT_PUBLIC_API_URL
│
└── README.md                           # This file
```

## Performance

### Latency (p95)
- **Total request**: ~2-3 seconds
  - Embedding (integrated): ~100ms
  - Vector search: ~200ms
  - Reranking: ~300ms
  - LLM generation: ~1-2s
  - Network overhead: ~200ms

### Cost (per 1000 requests)
- **Pinecone**: 
  - Search: ~$0.02 (serverless, 1M vectors)
  - Reranking: Free tier (10k/month)
- **OpenRouter (DeepSeek V3.1)**: Free tier
- **Tavily**: Free tier (1000 searches/month)
- **Total**: Effectively free on free tiers

### Memory Footprint
- **Backend**: ~200MB RAM (no local embeddings!)
- **Frontend**: Static site (negligible)

## How It Works

### Chunking Strategy

**Bible (Verse-Based)**
```python
# Group 3 consecutive verses
chunk = {
    "text": "verse1 verse2 verse3",
    "reference": "Philippians 4:6-8",
    "testament": "NT"  # or "OT"
}
```

**Harry Potter (Line-Based)**
```python
# Group 10 consecutive lines from CSV
chunk = {
    "text": "10 lines of narrative...",
    "reference": "Deathly Hallows, Chapter 33",
    "testament": "HP"
}
```

### Tradition-Aware Prompts

**Jewish**
```
"You are a compassionate Jewish guide. Write 2-4 paragraphs...
- Reference Torah/Tanakh verses
- Use Jewish concepts (mitzvot, tikkun olam)
- Focus on Hashem's love"
```

**Christian**
```
"You are a compassionate, non-denominational Christian guide...
- Reference Bible verses (OT/NT)
- Focus on God's love and grace
- Avoid theological jargon"
```

**Harry Potter**
```
"You are a compassionate guide who finds wisdom in stories...
- Draw parallels to themes (courage, friendship, loss)
- Reference characters and moments
- Avoid religious language"
```

**Social Media**
```
"You are a compassionate guide who finds wisdom in social media...
- Reference specific tweets and usernames
- Focus on real people's experiences
- Emphasize authenticity and relatability"
```

### Crisis Detection

Detects keywords like "suicide", "self-harm", "want to die" and immediately returns:
```
"I'm deeply concerned about what you're going through. 
Please reach out for immediate support:

• National Suicide Prevention Lifeline: 988 (24/7)
• Crisis Text Line: Text HOME to 741741
• International Association for Suicide Prevention: [link]

Your life has immeasurable value..."
```

## API Reference

### POST `/recommend/stream`

**Request**
```json
{
  "issue": "I'm feeling anxious about work",
  "tradition": "christian"  // "christian" | "jewish" | "harry_potter" | "social_media"
}
```

**Response (Streaming)**
```
data: {"type": "verses", "verses": [...]}
data: {"type": "explanation_chunk", "content": "..."}
data: {"type": "done"}
```

**Errors**
- `400`: Empty issue or > 500 characters
- `404`: No passages found
- `503`: Service not ready

### GET `/healthz`

**Response**
```json
{
  "ok": true,
  "db_verses": 37000,
  "framework": "Pinecone + DeepSeek",
  "reranker": "pinecone-rerank-v0"
}
```

## Deployment

### Backend (Render)
```bash
# render.yaml (auto-detected)
services:
  - type: web
    name: solace-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python main.py
    envVars:
      - key: PINECONE_API_KEY
        sync: false
      - key: OPENROUTER_API_KEY
        sync: false
      - key: TAVILY_API_KEY
        sync: false
```

### Frontend (Render Static Site)

```bash
# Render auto-detects Next.js
Build Command: npm install && npm run build
Publish Directory: frontend/out

# Environment variable
NEXT_PUBLIC_API_URL=https://your-backend.onrender.com
```

## Development

### Adding a New Source

1. **Create embedding script** (`data/embed_YOUR_SOURCE_pinecone.py`)
```python
chunk = {
    "text": "...",
    "reference": "Book, Chapter X",
    "testament": "YOUR_CODE",  # e.g., "LOTR"
    "translation": "Original"
}
```

2. **Update backend** (`backend/main.py`)
```python
elif request.tradition == "your_source":
    testament_filter = ["YOUR_CODE"]
```

3. **Add prompt**
```python
elif tradition == "your_source":
    system_prompt = """..."""
```

4. **Update frontend** (`frontend/app/page.js`)
```jsx
<option value="your_source">Your Source Name</option>
```

## Contributing

Ideas for future contributions:
- [ ] Add more sources (Lord of the Rings, Quran, Buddhist texts)
- [ ] Multi-turn conversation (agent mode)
- [ ] User accounts + saved passages
- [ ] Multilingual support
- [ ] Voice input
- [ ] Enhanced social media sharing features

## License

- **Code**: MIT License
- **Bible Text**: World English Bible (Public Domain)
- **Harry Potter Text**: Educational/transformative use, see [gastonstat/harry-potter-data](https://github.com/gastonstat/harry-potter-data)

## Acknowledgments

- [Pinecone](https://www.pinecone.io/) for serverless vector database + reranker
- [OpenRouter](https://openrouter.ai/) for LLM API access
- [DeepSeek](https://www.deepseek.com/) for the V3.1 model
- [Tavily](https://tavily.com/) for social media search
- [World English Bible](https://ebible.org/web/) translators
- [gastonstat](https://github.com/gastonstat) for Harry Potter dataset
- [LangSmith](https://smith.langchain.com/) for tracing tools

---

**Built to provide comfort and encouragement through the texts you love.**

*"Happiness can be found, even in the darkest of times, if one only remembers to turn on the light." — Albus Dumbledore*

---

Made by **[Parakh Jaggi](https://www.linkedin.com/in/parakhjaggi/)**