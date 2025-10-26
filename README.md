# 📖 Bible Verse Companion (MVP)

A semantic search application that helps users find comforting Bible verses based on their emotional or spiritual concerns. Built with state-of-the-art AI embeddings using [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B).

## 🎯 Core Value Proposition

**User types what they're going through → gets 1-3 highly relevant verses + a short, empathetic explanation.**

### User Stories

1. **"I'm anxious about work."** → Relevant verses + 120–180-word encouragement
2. **"I feel guilty about a mistake."** → Verses on confession/forgiveness + why they help
3. **"I'm feeling lonely."** → Comforting verses + empathetic context

## 🌟 Features

- ✅ Semantic search using Qwen3-Embedding-8B (state-of-the-art multilingual model)
- ✅ Vector database with ChromaDB for fast similarity search
- ✅ World English Bible (complete Bible text)
- ✅ Persistent database for production deployment
- ✅ Mobile-first design (planned)
- ✅ Faith-based encouragement for real-world concerns

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Create the Vector Database

```bash
cd data
python embed_bible_qwen.py
```

This will:
- Parse the Bible XML file (engwebp_vpl.xml)
- Create embeddings using Qwen3-Embedding-8B
- Store in ChromaDB for fast retrieval
- Takes ~10-30 minutes on CPU (3-10 minutes on GPU)

### Try It Out

```bash
python example_query.py
```

This runs example queries and starts an interactive mode where you can type your concerns.

## 📚 Documentation

- **[SETUP.md](SETUP.md)** - Detailed setup instructions, troubleshooting, and production usage
- **[data/embed_bible_qwen.py](data/embed_bible_qwen.py)** - Database creation script
- **[data/example_query.py](data/example_query.py)** - Example query implementation

## 🏗️ Architecture

```
┌─────────────────┐
│  User Input     │  "I'm anxious about work"
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Qwen3-Embedding-8B                 │  Convert query to vector
│  + Query Instruction                │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  ChromaDB Vector Search             │  Find similar verses
│  (Cosine Similarity)                │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Top 1-3 Most Relevant Verses       │  Philippians 4:6-8, etc.
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  LLM-Generated Explanation          │  120-180 words (Future)
│  (Empathetic Context)               │
└─────────────────────────────────────┘
```

## 🤖 Technology Stack

- **Embedding Model:** [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B)
  - 8B parameters, 4096 dimensions
  - #1 on MTEB multilingual leaderboard
  - 100+ language support
  
- **Vector Database:** ChromaDB
  - Fast similarity search
  - Persistent storage
  - Easy production deployment

- **Bible Source:** World English Bible (Public Domain)
  - Complete Old and New Testament
  - Modern English translation

## 📊 Project Structure

```
ask-book/
├── data/
│   ├── engwebp_vpl.xml              # Bible XML source (31K+ verses)
│   ├── embed_bible_qwen.py          # Database creation script ⭐
│   ├── example_query.py             # Query example ⭐
│   └── output_bible_db_qwen/        # Generated vector DB (after setup)
├── requirements.txt                  # Python dependencies
├── README.md                        # This file
└── SETUP.md                         # Detailed setup guide
```

## 🎯 Roadmap

### ✅ Phase 1: Core Vector Search (Current)
- [x] XML parsing
- [x] Qwen3-Embedding-8B integration
- [x] ChromaDB vector storage
- [x] Basic query interface

### 🚧 Phase 2: Backend API (Next)
- [ ] FastAPI REST API
- [ ] Query endpoint with verse retrieval
- [ ] LLM integration for explanations (120-180 words)
- [ ] Rate limiting and caching

### 📱 Phase 3: Mobile-First Frontend
- [ ] React/React Native UI
- [ ] Clean, empathetic design
- [ ] Input field + verse display
- [ ] Save favorite verses

### 🚀 Phase 4: Deployment
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/GCP)
- [ ] CDN for static assets
- [ ] Monitoring and analytics

## 🔍 How It Works

### The Query Instruction

The key to good semantic search is the **query instruction**:

```
Represent the emotional or spiritual concern described by the user 
to retrieve comforting Bible passages:
```

This instruction tells the embedding model to:
1. Interpret user input as emotional/spiritual concerns
2. Match with Bible verses that provide comfort
3. Prioritize relevance for encouragement

### Example Queries

| User Input | Matches Verses About |
|------------|---------------------|
| "I'm anxious" | Peace, trust in God, casting worries |
| "I feel guilty" | Forgiveness, confession, God's mercy |
| "I'm lonely" | God's presence, community, comfort |
| "I need courage" | Strength, faith, overcoming fear |

## 🛠️ Production Usage

### Loading the Database

```python
import chromadb
from sentence_transformers import SentenceTransformer

# Load database
client = chromadb.PersistentClient(path="./data/output_bible_db_qwen")
collection = client.get_collection("bible_verses")

# Load model
model = SentenceTransformer("Qwen/Qwen3-Embedding-8B", trust_remote_code=True)
```

### Querying

```python
# User concern
query = "I'm feeling anxious"

# Embed query with instruction
instruction = "Represent the emotional or spiritual concern..."
embedding = model.encode([f"{instruction} {query}"])

# Search
results = collection.query(query_embeddings=embedding.tolist(), n_results=3)

# Access results
for i, verse_text in enumerate(results['documents'][0]):
    reference = results['metadatas'][0][i]['reference']
    print(f"{reference}: {verse_text}")
```

See [SETUP.md](SETUP.md) for complete documentation.

## 📈 Performance

- **Embedding creation:** 10-30 min (CPU) / 3-10 min (GPU)
- **Query time:** <100ms per query (with model loaded)
- **Database size:** ~500MB-1GB
- **Memory usage:** ~8GB during embedding, ~2GB during queries

## 🤝 Contributing

This is an MVP project. Future contributions could include:
- Additional Bible translations
- Multilingual support (leveraging Qwen3's 100+ languages)
- Alternative embedding models
- Query result ranking improvements
- Frontend development

## 📄 License

- **Code:** MIT License (or your choice)
- **Bible Text:** World English Bible (Public Domain)
- **Model:** Qwen3-Embedding-8B (Apache 2.0)

## 🙏 Acknowledgments

- [Qwen Team](https://huggingface.co/Qwen) for the excellent Qwen3-Embedding models
- World English Bible translators
- ChromaDB team for the vector database
- HuggingFace for model hosting and tooling

## 📧 Contact

For questions or suggestions, please open an issue.

---

**Built with ❤️ to provide comfort and encouragement through faith.**
