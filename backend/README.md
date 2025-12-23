# Document Embedding API

A FastAPI backend service that processes documents (PDF, TXT, MD, DOCX) and generates sentence-level embeddings with paragraph tracking using HuggingFace sentence-transformers, NLTK, and FAISS for semantic search.

## Features

- 📄 Support for multiple document formats: PDF, TXT, MD, DOCX
- 🤖 HuggingFace embeddings using sentence-transformers
- 📝 Sentence-level embedding with paragraph tracking
- 🔍 FAISS-based semantic search across indexed documents
- 💾 Persistent vector store with metadata
- 🚀 Fast and efficient batch embedding generation
- 🔒 CORS enabled for frontend integration

## Setup

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Server

Start the development server:
```bash
uvicorn app.main:app --reload
```

Or run directly:
```bash
python -m app.main
```

The API will be available at `http://localhost:8000`

### Testing

Create a test document `test.txt` with sample text and test the endpoints:

**Test basic embedding (without indexing):**
```bash
curl -X POST "http://localhost:8000/embed" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test.txt"
```

**Test indexing and search:**

```bash
# 1. Index a document
curl -X POST "http://localhost:8000/index" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test.txt"

# 2. Search indexed documents
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "what is consciousness?", "top_k": 3}'

# 3. List all indexed documents
curl -X GET "http://localhost:8000/documents"

# 4. Get vector store statistics
curl -X GET "http://localhost:8000/stats"

# 5. Delete a document (replace {doc_id} with actual ID from step 1)
curl -X DELETE "http://localhost:8000/documents/{doc_id}"
```

**Test with multiple documents:**
```bash
# Create another test file
echo "Epistemology studies the nature of knowledge." > test2.txt

# Index both documents
curl -X POST "http://localhost:8000/index" -F "file=@test.txt"
curl -X POST "http://localhost:8000/index" -F "file=@test2.txt"

# Search across all documents
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "knowledge", "top_k": 5}'
```

Or test with Python:
```python
from app.utils.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService

doc_processor = DocumentProcessor()
embedding_service = EmbeddingService()

test_text = """Your test text here. With multiple sentences.

And multiple paragraphs too."""

sentences, para_indices, embeddings = embedding_service.embed_by_sentence(test_text, doc_processor)
print(f"Sentences: {len(sentences)}, Paragraphs: {max(para_indices) + 1}")
```

## API Endpoints

### Health Check
```
GET /
GET /health
```
Check if the API is running.

### Document Embedding
```
POST /embed
```
Upload a document and get sentence-level embeddings with paragraph tracking.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: File upload (field name: `file`)

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/embed" \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "filename": "document.pdf",
  "file_type": ".pdf",
  "text_length": 1234,
  "sentence_count": 15,
  "paragraph_count": 5,
  "sentences": [
    "First sentence.",
    "Second sentence.",
    "Third sentence in new paragraph.",
    ...
  ],
  "paragraph_indices": [0, 0, 1, ...],
  "embeddings": [
    [0.123, -0.456, 0.789, ...],
    [0.234, -0.567, 0.890, ...],
    ...
  ],
  "embedding_dimension": 384
}
```

**Response Fields:**
- `sentences`: List of extracted sentences
- `paragraph_indices`: For each sentence, which paragraph it belongs to (0-indexed)
- `embeddings`: Embedding vector for each sentence
- `sentence_count`: Total number of sentences
- `paragraph_count`: Total number of paragraphs detected

---

### Index Document
```
POST /index
```
Upload and index a document for semantic search. Documents are stored in the FAISS vector store for later searching.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: File upload (field name: `file`)

**Example:**
```bash
curl -X POST "http://localhost:8000/index" \
  -F "file=@philosophy.pdf"
```

**Response:**
```json
{
  "doc_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "philosophy.pdf",
  "file_type": ".pdf",
  "sentence_count": 42,
  "paragraph_count": 12,
  "message": "Document indexed successfully"
}
```

---

### Search Documents
```
POST /search
```
Search indexed documents for semantically similar passages. Returns matched sentences grouped by paragraph.

**Request:**
- Method: `POST`
- Content-Type: `application/json`
- Body: JSON with search parameters

**Request Body:**
```json
{
  "query": "what is consciousness?",
  "top_k": 5,
  "deduplicate_paragraphs": true
}
```

**Parameters:**
- `query` (required): Search query text
- `top_k` (optional, default: 5): Number of results to return
- `deduplicate_paragraphs` (optional, default: true): Return only one result per paragraph

**Example:**
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "what is consciousness?", "top_k": 3}'
```

**Response:**
```json
{
  "query": "what is consciousness?",
  "total_results": 3,
  "results": [
    {
      "doc_id": "550e8400-e29b-41d4-a716-446655440000",
      "filename": "philosophy.pdf",
      "paragraph_index": 2,
      "paragraph_text": "The question of consciousness has perplexed philosophers for centuries. What is it that makes us aware of our own existence?",
      "matched_sentences": [
        "The question of consciousness has perplexed philosophers for centuries.",
        "What is it that makes us aware of our own existence?"
      ],
      "similarity_scores": [0.89, 0.85]
    }
  ]
}
```

---

### List Documents
```
GET /documents
```
List all indexed documents in the vector store.

**Example:**
```bash
curl -X GET "http://localhost:8000/documents"
```

**Response:**
```json
{
  "total_documents": 3,
  "documents": [
    {
      "doc_id": "550e8400-e29b-41d4-a716-446655440000",
      "filename": "philosophy.pdf",
      "file_type": ".pdf",
      "total_sentences": 42,
      "total_paragraphs": 12
    }
  ]
}
```

---

### Delete Document
```
DELETE /documents/{doc_id}
```
Remove a document from the vector store index.

**Parameters:**
- `doc_id`: Document ID (from `/documents` or `/index` response)

**Example:**
```bash
curl -X DELETE "http://localhost:8000/documents/550e8400-e29b-41d4-a716-446655440000"
```

**Response:**
```json
{
  "message": "Document deleted successfully",
  "doc_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

### Get Statistics
```
GET /stats
```
Get statistics about the vector store.

**Example:**
```bash
curl -X GET "http://localhost:8000/stats"
```

**Response:**
```json
{
  "total_documents": 5,
  "total_sentences": 237,
  "total_paragraphs": 89,
  "embedding_dimension": 384
}
```

---

## Supported File Types

- **PDF** (`.pdf`) - Extracts text from all pages
- **Text** (`.txt`) - Plain text files
- **Markdown** (`.md`) - Markdown documents
- **Word** (`.docx`) - Microsoft Word documents

## Text Processing

The API uses NLTK for sentence tokenization and intelligently detects paragraph boundaries:
- Paragraphs are split by double newlines (`\n\n`) when available
- Falls back to single newlines (`\n`) for documents without double-newline formatting
- Empty sentences and paragraphs are automatically filtered out
- Paragraph indices are sequential (0, 1, 2...) with no gaps

## Embedding Model

By default, the API uses the `all-MiniLM-L6-v2` model from sentence-transformers, which provides:
- Fast inference
- 384-dimensional embeddings
- Good quality for most use cases

To use a different model, modify the `EmbeddingService` initialization in `app/services/embedding_service.py`:

```python
# For higher quality (but slower):
embedding_service = EmbeddingService(model_name="all-mpnet-base-v2")

# For multilingual support:
embedding_service = EmbeddingService(model_name="paraphrase-multilingual-MiniLM-L12-v2")
```

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application & endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── embedding_service.py   # Sentence-level embedding generation
│   │   └── vector_store.py        # FAISS vector store management
│   └── utils/
│       ├── __init__.py
│       └── document_processor.py  # Document extraction & sentence splitting
├── data/                           # FAISS index & metadata (persistent storage)
├── uploads/                        # Temporary file storage (auto-created)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Vector Store

The API uses FAISS (Facebook AI Similarity Search) for efficient semantic search:
- **Index**: L2 distance-based similarity
- **Storage**: Persistent storage in `data/` directory
- **Metadata**: Document info, sentences, and paragraph mappings stored in JSON
- **Search**: Sentence-level search with automatic paragraph grouping and deduplication

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Error Handling

The API returns appropriate HTTP status codes:
- `200` - Success
- `400` - Bad request (unsupported file type, empty document, etc.)
- `404` - Not found (document doesn't exist)
- `500` - Internal server error

Error responses include a `detail` field with the error message.

## Notes

- Uploaded files are temporarily saved and deleted after processing
- The first request may be slower as the model and NLTK data load into memory
- Embeddings are deterministic for the same input text
- Sentence splitting uses NLTK's punkt tokenizer (downloads automatically on first use)
- FAISS index and metadata are persisted to disk in the `data/` directory
- On Apple Silicon, the CPU version of FAISS is used and performs well
