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
| `/index-distributed` | POST | Index document (Ray distributed, recommended for 1000+ sentences) |
| `/search` | POST | Semantic search with optional reranking |
| `/documents` | GET | List indexed documents |
| `/documents/{doc_id}` | DELETE | Remove document |
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

## 2-Stage Retrieval Architecture

Production-quality search combining FAISS approximate search with cross-encoder reranking.

### Pipeline

```
Query → FAISS (top-50) → Group by Paragraph → Cross-Encoder Rerank (top-10) → Context Expansion → Results
```

**Stage 1 (FAISS):** all-MiniLM-L6-v2 bi-encoder retrieves ~50 candidate sentences (~0.05s)  
**Stage 2 (Reranker):** BAAI/bge-reranker-v2-m3 cross-encoder reranks to top-10 paragraphs (~0.31s)  
**Context:** Optionally expand with ±N paragraphs for reading context

### Performance Benchmarks

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
