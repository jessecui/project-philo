# Project Philo

A semantic search and RAG (Retrieval-Augmented Generation) platform for philosophical texts, featuring a Vite + React frontend and FastAPI backend with 2-stage retrieval and AI-powered answers.

## Features

- 🔍 **Semantic Search**: Sentence embeddings with all-MiniLM-L6-v2 (384-dim)
- 🎯 **2-Stage Retrieval**: FAISS + cross-encoder reranking (88% vs 62% accuracy, +26pp improvement)
- 🤖 **RAG Generation**: Gemini 3 Flash streaming answers with source citations
- 📄 **Multi-format Support**: PDF, TXT, MD, DOCX
- ⚡ **Ray Distributed Processing**: 3.96x speedup on 10 cores
- 🚀 **Automatic Device Detection**: MPS/CUDA/CPU

## Tech Stack

- **Frontend**: Vite, React 19, TypeScript, Tailwind CSS v4
- **Backend**: FastAPI, Python
- **Vector Store**: FAISS with persistent storage
- **AI/ML**:
  - Embeddings: all-MiniLM-L6-v2 (384-dim, bi-encoder)
  - Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Generation: Gemini 3 Flash (1M token context, streaming)
- **Distributed Processing**: Ray

---

## Quick Start

### Prerequisites

- Node.js 18+
- Python 3.10+
- Google Cloud account (for Gemini API)

### 1. Clone & Install

```bash
git clone https://github.com/jessecui/project-philo.git
cd project-philo

# Frontend dependencies
cd frontend
npm install
cd ..

# Backend dependencies
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment

Create `backend/.env`:

```bash
GOOGLE_API_KEY=your-google-ai-api-key
CREATOR_NAME=your-name  # For auth modal
# Optional: ENABLE_RAY=true  # For distributed processing
```

### 3. Index the Texts

Index the philosophical texts in `backend/texts/` to create the FAISS vector store:

```bash
# From backend/ directory
python -m app.scripts.index_texts
```

This will generate `backend/data/faiss.index` and `backend/data/metadata.json`.

### 4. Run the Application

```bash
# Terminal 1: Start backend (from backend/)
uvicorn app.main:app --reload
# API at http://localhost:8000

# Terminal 2: Start frontend (from frontend/)
npm run dev
# App at http://localhost:5173
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/index` | POST | Index document (single-threaded) |
| `/index-distributed` | POST | Index document (Ray distributed) |
| `/search` | POST | Semantic search with optional reranking |
| `/search-and-generate` | POST | RAG: Retrieve + Generate streaming answer |
| `/documents` | GET | List indexed documents |
| `/documents/{doc_id}` | DELETE | Remove document |
| `/stats` | GET | Vector store statistics |

### Search Parameters

- `query` (required): Search text
- `use_reranking` (default: false): Enable 2-stage retrieval
- `top_k` (default: 5): Results to return
- `top_k_faiss` (default: 30): FAISS candidates for reranking
- `context_window` (default: 2): Paragraphs before/after for context

---

## Usage Examples

### Index a Document

```bash
curl -X POST "http://localhost:8000/index" \
  -F "file=@document.pdf"
```

### Search with Reranking

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what is consciousness?",
    "use_reranking": true,
    "top_k": 10,
    "context_window": 2
  }'
```

### RAG: Search & Generate Answer

```bash
curl -N -X POST "http://localhost:8000/search-and-generate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the nature of the Tao?",
    "top_k_context": 5,
    "use_reranking": true,
    "temperature": 0.7
  }'
```

### Test Scripts

```bash
# Test retrieval pipeline
python -m app.scripts.test_search

# Test RAG pipeline with pretty output
python -m app.scripts.test_search_and_generate "How should one cultivate virtue?"
```

---

## RAG Response Format

Server-Sent Events (SSE) streaming:

```json
// Initial: Retrieved sources
data: {"type":"sources","data":[{"filename":"Tao_Te_Ching.txt","paragraph_idx":5,"text":"...","score":0.85}]}

// Streaming: Tokens as they're generated
data: {"type":"token","data":"The "}
data: {"type":"token","data":"Tao "}

// Final: Timing metrics
data: {"type":"done","data":{"generation_time":2.34,"total_time":2.52}}
```

---

## Performance Benchmarks

### Retrieval Quality

Tested on 50 philosophical queries with 100 paragraphs:

| Metric | FAISS-only | FAISS + Reranking | Improvement |
|--------|-----------|------------------|-------------|
| **Accuracy@1** | 62% | 88% | **+26pp** |
| **nDCG@10** | 0.760 | 0.899 | **+18.3%** |
| **MRR** | 0.740 | 0.930 | **+25.6%** |
| **Query Time** | ~5ms | ~50ms | 10x slower |

### When to Use Reranking

**Use `use_reranking=true` for:**
- User-facing search (quality critical)
- Top-k precision requirements
- Query time < 500ms acceptable

**Use `use_reranking=false` for:**
- Real-time applications (< 50ms)
- Broad recall needed
- Resource-constrained environments

### Ray Distributed Processing

Tested with 19,900 sentences (10 Ray workers on 12-core Mac):

| Method | Time | Throughput | Speedup |
|--------|------|------------|---------|
| **Sequential (1 core)** | 29.18s | 682 sent/s | 1.0x |
| **Ray (10 cores)** | 7.36s | 2,704 sent/s | **3.96x** |

**Recommendation:** Use `/index-distributed` for documents with 1000+ sentences.

### Run Evaluations

```bash
# Test retrieval quality (FAISS vs FAISS+reranking)
python -m app.evaluation.eval_faiss_cross_encoder_ndcg

# Test ingestion performance (sequential vs Ray distributed)
python -m app.evaluation.eval_ray_ingestion_latency
```

---

## Project Structure

```
project-philo/
├── frontend/src/                       # Vite + React frontend
│   ├── app/
│   │   ├── page.tsx                    # Main search interface
│   │   ├── layout.tsx                  # App layout
│   │   └── globals.css                 # Global styles
│   ├── components/
│   │   ├── auth-modal.tsx              # Authentication modal
│   │   └── ui/                         # UI components
│   └── lib/
│       └── utils.ts                    # Utility functions
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI endpoints
│   │   ├── services/
│   │   │   ├── embedding_service.py    # Sentence embeddings
│   │   │   ├── reranker_service.py     # Cross-encoder reranking
│   │   │   ├── vector_store.py         # FAISS vector store
│   │   │   ├── generation_service.py   # Gemini 3 Flash generation
│   │   │   └── distributed_ingestion.py # Ray parallel processing
│   │   ├── utils/
│   │   │   └── document_processor.py   # Text extraction & splitting
│   │   ├── scripts/
│   │   │   ├── index_texts.py          # Build FAISS index from texts/
│   │   │   ├── test_search.py          # Test retrieval pipeline
│   │   │   └── test_search_and_generate.py # Test RAG pipeline
│   │   └── evaluation/
│   │       ├── eval_faiss_cross_encoder_ndcg.py
│   │       └── eval_ray_ingestion_latency.py
│   ├── data/                           # FAISS index & metadata
│   ├── texts/                          # Sample philosophical texts
│   └── requirements.txt
├── package.json
└── README.md
```

---

## Technical Details

**Device Support:**
- Auto-detects MPS (Apple Silicon) / CUDA (NVIDIA) / CPU
- Both embedding and reranker models use same device

**Storage:**
- FAISS index: `backend/data/faiss.index`
- Metadata: `backend/data/metadata.json`
- Persistent across restarts

**Text Processing:**
- Sentence tokenization: NLTK punkt
- Paragraph detection: double newlines (`\n\n`)
- Automatic filtering of empty content

**Pricing (Gemini 3 Flash):**
- ~$1.25/1M input tokens, ~$5.00/1M output tokens
- ~$0.003-0.005 per query

---

## Troubleshooting

**"Gemini generator not initialized"**
- Check `backend/.env` file exists with `GOOGLE_API_KEY`
- Get API key from https://aistudio.google.com/apikey

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
