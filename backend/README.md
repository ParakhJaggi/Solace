# Backend - Bible Verse Companion API

FastAPI backend for semantic Bible verse search.

## Quick Start

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Run the server

```bash
python main.py
```

Server will start on `http://localhost:8080`

### 3. Test it

In another terminal:

```bash
python test_api.py
```

Or use the interactive docs at: `http://localhost:8080/docs`

## Endpoints

### `GET /healthz`
Health check - returns status and model info

### `POST /recommend`
Get verse recommendations

**Request:**
```json
{
  "issue": "I'm feeling anxious about work",
  "k": 20,
  "n": 3
}
```

**Response:**
```json
{
  "verses": [
    {
      "ref": "Philippians 4:6-8",
      "text": "Don't be anxious for anything...",
      "translation": "WEB",
      "score": 0.892
    }
  ],
  "explanation": ""
}
```

## Current Status

**✅ Step 1 Complete:**
- Load ChromaDB at startup
- Embed user queries with Qwen3-Embedding-0.6B
- Return top N verses

**🚧 Next Steps:**
- Add reranker (cross-encoder)
- Add LLM for explanations
- Add crisis detection

## Environment Variables

```bash
CHROMA_DIR=../data/output_bible_db_qwen
EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B
QUERY_INSTRUCTION="Represent the emotional or spiritual concern..."
PORT=8080
```

## Manual Testing with curl

```bash
# Health check
curl http://localhost:8080/healthz

# Get recommendations
curl -X POST http://localhost:8080/recommend \
  -H "Content-Type: application/json" \
  -d '{"issue": "I am feeling anxious", "n": 3}'
```

