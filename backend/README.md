# Document Embedding API

A FastAPI backend service that processes documents (PDF, TXT, MD, DOCX) and generates sentence-level embeddings with paragraph tracking using HuggingFace sentence-transformers and NLTK.

## Features

- 📄 Support for multiple document formats: PDF, TXT, MD, DOCX
- 🤖 HuggingFace embeddings using sentence-transformers
- 📝 Sentence-level embedding with paragraph tracking
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

Create a test document `test.txt` with sample text and test the embedding endpoint:

```bash
curl -X POST "http://localhost:8000/embed" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test.txt"
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
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/document.pdf"
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
│   ├── main.py                    # FastAPI application
│   ├── services/
│   │   ├── __init__.py
│   │   └── embedding_service.py   # Sentence-level embedding generation
│   └── utils/
│       ├── __init__.py
│       └── document_processor.py  # Document extraction & sentence splitting
├── uploads/                        # Temporary file storage (auto-created)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Error Handling

The API returns appropriate HTTP status codes:
- `200` - Success
- `400` - Bad request (unsupported file type, empty document, etc.)
- `500` - Internal server error

Error responses include a `detail` field with the error message.

## Notes

- Uploaded files are temporarily saved and deleted after processing
- The first request may be slower as the model and NLTK data load into memory
- Embeddings are deterministic for the same input text
- Sentence splitting uses NLTK's punkt tokenizer (downloads automatically on first use)
