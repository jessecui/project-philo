# Document Embedding API

A FastAPI service for document processing and semantic search using sentence-level embeddings with FAISS vector store and 2-stage retrieval with cross-encoder reranking.

## Features

- 📄 Multi-format support: PDF, TXT, MD, DOCX
- 🤖 Sentence embeddings (all-MiniLM-L6-v2, 384-dim) with paragraph tracking
- 🔍 FAISS vector store with persistent storage
- 🎯 **2-stage retrieval: FAISS + cross-encoder reranking (94% vs 62% accuracy, +32pp improvement)**
- 📊 Context window expansion for richer results
- ⚡ Ray distributed processing (3.96x speedup on 10 cores)
- 🚀 Automatic MPS/CUDA/CPU detection
- 🤖 **RAG Generation: Gemini 2.5 Pro streaming answers with source citations**

## Quick Start

```bash
# Install dependencies
cd backend
pip install -r requirements.txt

# Run server
uvicorn app.main:app --reload
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Basic Usage

```bash
# Index a document
curl -X POST "http://localhost:8000/index" \
  -F "file=@document.pdf"

# Search with reranking (high quality)
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what is consciousness?",
    "use_reranking": true,
    "top_k": 10,
    "context_window": 2
  }'

# Search without reranking (fast)
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "consciousness", "top_k": 5}'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/index` | POST | Index document (single-threaded) |
| `/search` | POST | Semantic search with optional reranking |
| `/search-and-generate` | POST | RAG: Retrieve + Generate streaming answer |
| `/documents` | GET | List indexed documents |
| `/documents/{doc_id}` | DELETE | Remove document |
| `/stats` | GET | Vector store statistics |cument |
| `/stats` | GET | Vector store statistics |

**Search Parameters:**
- `query` (required): Search text
- `use_reranking` (default: false): Enable 2-stage retrieval
- `top_k` (default: 5): Results to return
- `top_k_faiss` (default: 50): FAISS candidates for reranking
- `context_window` (default: 2): Paragraphs before/after for context
- `deduplicate_paragraphs` (default: true): One result per paragraph
**Supported Formats:** PDF, TXT, MD, DOCX

---

## RAG (Retrieval-Augmented Generation)

Generate AI-powered answers grounded in your indexed documents using Gemini 2.5 Pro.

### Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt  # Includes google-genai
```

2. **Configure Google Cloud credentials:**
```bash
cp .env.example .env
# Edit .env with your project details
```

Required environment variables:
- `GOOGLE_CLOUD_PROJECT` - Your GCP project ID
- `VERTEX_AI_LOCATION` - Region (e.g., us-central1)
- `GOOGLE_APPLICATION_CREDENTIALS` - Path to service account key (optional if using ADC)

3. **Enable Vertex AI API:**
```bash
gcloud services enable aiplatform.googleapis.com
```

Or use Application Default Credentials for local development:
```bash
gcloud auth application-default login
```

### Usage

**Streaming Generation (recommended):**
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

**Test Script with Pretty Output:**
```bash
# Default query
python -m app.scripts.test_search_and_generate

# Custom query
python -m app.scripts.test_search_and_generate "How should one cultivate virtue?"

# With parameters
python -m app.scripts.test_search_and_generate \
  "What is self-reliance?" \
  --top-n-context 3 \
  --temperature 0.5
```

**Parameters:**
- `query` (required): User's question
- `top_k_context` (default: 5): Number of excerpts to retrieve
- `use_reranking` (default: true): Use cross-encoder for better quality
- `temperature` (default: 0.7): Generation randomness (0.0-1.0)
- `max_output_tokens` (default: 2048): Maximum response length

**Response Format (Server-Sent Events):**
```json
// Initial: Retrieved sources
data: {"type":"sources","data":[{"filename":"Tao_Te_Ching.txt","paragraph_idx":5,"text":"...","score":0.85}]}

// Streaming: Tokens as they're generated
data: {"type":"token","data":"The "}
data: {"type":"token","data":"Tao "}

// Final: Timing metrics
data: {"type":"done","data":{"generation_time":2.34,"total_time":2.52}}
```
**Models:**
- Embeddings: all-MiniLM-L6-v2 (384-dim, bi-encoder)
- Reranker: BAAI/bge-reranker-v2-m3 (cross-encoder)
- Generation: Gemini 2.5 Pro (1M token context, streaming)
- **Model:** Gemini 2.5 Pro
- **Context Window:** 1M tokens
- **Features:** Adaptive thinking, grounded generation, source citations
- **Pricing:** ~$1.25/1M input tokens, ~$5.00/1M output tokens (~$0.003-0.005 per query)

### Troubleshooting

**"Vertex AI generator not initialized"**
- Check `.env` file exists and has correct values
- Verify `GOOGLE_CLOUD_PROJECT` is set
- Ensure Vertex AI API is enabled: `gcloud services list --enabled | grep aiplatform`
- Try: `gcloud auth application-default login`

**Permission denied errors**
- Service account needs `Vertex AI User` role (`roles/aiplatform.user`)

---
---

backend/
├── app/
│   ├── main.py                        # FastAPI endpoints
│   ├── services/
│   │   ├── embedding_service.py       # Sentence embeddings
│   │   ├── reranker_service.py        # Cross-encoder reranking
│   │   ├── vector_store.py            # FAISS vector store
│   │   ├── generation_service.py      # Gemini 2.5 Pro generation
│   │   └── distributed_ingestion.py   # Ray parallel processing
│   ├── utils/
│   │   └── document_processor.py      # Text extraction & splitting
│   ├── scripts/
│   │   ├── test_search.py                    # Test retrieval pipeline
│   │   └── test_search_and_generate.py       # Test RAG pipeline
│   └── evaluation/
│       ├── eval_faiss_cross_encoder_ndcg.py  # Retrieval quality tests
│       └── eval_ray_ingestion_latency.py     # Performance benchmarks
├── data/                               # FAISS index & metadata
├── .env.example                        # Environment variable template
└── requirements.txthmarks

Tested on 50 philosophical queries with 100 paragraphs (50 utilitarianism variants + 50 other ethical theories):

| Metric | FAISS-only | FAISS + Reranking | Improvement |
|--------|-----------|------------------|-------------|
| **Accuracy@1** | 62% | 94% | **+32pp** |
| **nDCG@10** | 0.760 | 0.928 | **+22.1%** |
| **MRR** | 0.740 | 0.965 | **+30.4%** |
| **Query Time** | ~50ms | ~360ms | 7x slower |

### When to Use Reranking

**Use `use_reranking=true` for:**
- User-facing search (quality critical)
- Top-k precision requirements
- Query time < 500ms acceptable

**Use `use_reranking=false` for:**
- Real-time applications (< 50ms)
- Broad recall needed
- Resource-constrained environments

### Run Evaluations

```bash
# Test retrieval quality (FAISS vs FAISS+reranking) on 50 test queries
python -m app.evaluation.eval_faiss_cross_encoder_ndcg

# Test ingestion performance (sequential vs Ray distributed)
python -m app.evaluation.eval_ray_ingestion_latency
```

---

## Ray Distributed Processing

Parallelize embedding generation across CPU cores using Ray.

### Performance

Tested with 19,900 sentences (10 Ray workers on 12-core Mac):

| Method | Time | Throughput | Speedup |
|--------|------|------------|---------|
| **Sequential (1 core)** | 29.18s | 682 sent/s | 1.0x |
| **Ray (10 cores)** | 7.36s | 2,704 sent/s | **3.96x** |

*Excluding 6.35s initialization time for Ray workers*

### Usage

```bash
# Sequential indexing (small documents)
curl -X POST "http://localhost:8000/index" -F "file=@doc.pdf"

# Distributed indexing (1000+ sentences)
curl -X POST "http://localhost:8000/index-distributed" -F "file=@large_doc.pdf"
```

**Recommendation:** Use distributed indexing for documents with 1000+ sentences.

---

## Technical Details

**Models:**
- Embeddings: all-MiniLM-L6-v2 (384-dim, bi-encoder)
- Reranker: BAAI/bge-reranker-v2-m3 (cross-encoder)

**Device Support:**
- Auto-detects MPS (Apple Silicon) / CUDA (NVIDIA) / CPU
- Both models use same device for consistency

**Storage:**
- FAISS index: `data/faiss.index`
- Metadata: `data/metadata.json`
- Persistent across restarts

**Text Processing:**
- Sentence tokenization: NLTK punkt
- Paragraph detection: double newlines (`\n\n`)
- Automatic filtering of empty content

---

## Project Structure

```
backend/
├── app/
│   ├── main.py                        # FastAPI endpoints
│   ├── services/
│   │   ├── embedding_service.py       # Sentence embeddings
│   │   ├── reranker_service.py        # Cross-encoder reranking
│   │   ├── vector_store.py            # FAISS vector store
│   │   └── distributed_ingestion.py   # Ray parallel processing
│   ├── utils/
│   │   └── document_processor.py      # Text extraction & splitting
│   └── evaluation/
│       ├── eval_faiss_cross_encoder_ndcg.py  # Retrieval quality tests
│       └── eval_ray_ingestion_latency.py     # Performance benchmarks
├── data/                               # FAISS index & metadata
└── requirements.txt
```

---

## API Documentation

Interactive docs: `http://localhost:8000/docs`
