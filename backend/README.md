# Document Embedding API

A FastAPI backend service that processes documents (PDF, TXT, MD, DOCX) and generates embeddings using HuggingFace sentence-transformers.

## Features

- 📄 Support for multiple document formats: PDF, TXT, MD, DOCX
- 🤖 HuggingFace embeddings using sentence-transformers
- 🚀 Fast and efficient embedding generation
- 📦 Batch processing support for multiple documents
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

## API Endpoints

### Health Check
```
GET /
GET /health
```
Check if the API is running.

### Single Document Embedding
```
POST /embed
```
Upload a single document and get its embedding.

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
  "text_preview": "First 200 characters of extracted text...",
  "embedding_dimension": 384,
  "embedding": [0.123, -0.456, 0.789, ...]
}
```

### Batch Document Embedding
```
POST /embed-batch
```
Upload multiple documents (up to 10) and get their embeddings.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Multiple file uploads (field name: `files`)

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/embed-batch" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@document1.pdf" \
  -F "files=@document2.txt" \
  -F "files=@document3.md"
```

**Response:**
```json
{
  "results": [
    {
      "filename": "document1.pdf",
      "file_type": ".pdf",
      "text_length": 1234,
      "text_preview": "First 200 characters...",
      "embedding_dimension": 384,
      "embedding": [...]
    },
    {
      "filename": "document2.txt",
      "file_type": ".txt",
      "text_length": 567,
      "text_preview": "First 200 characters...",
      "embedding_dimension": 384,
      "embedding": [...]
    }
  ]
}
```

## Supported File Types

- **PDF** (`.pdf`) - Extracts text from all pages
- **Text** (`.txt`) - Plain text files
- **Markdown** (`.md`) - Markdown documents
- **Word** (`.docx`) - Microsoft Word documents

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
│   │   └── embedding_service.py   # Embedding generation
│   └── utils/
│       ├── __init__.py
│       └── document_processor.py  # Document text extraction
├── uploads/                        # Temporary file storage (auto-created)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Development

### Testing with Python
```python
import requests

# Test embedding endpoint
with open('test.pdf', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/embed', files=files)
    print(response.json())
```

## Error Handling

The API returns appropriate HTTP status codes:
- `200` - Success
- `400` - Bad request (unsupported file type, empty document, etc.)
- `500` - Internal server error

Error responses include a `detail` field with the error message.

## Notes

- Uploaded files are temporarily saved and deleted after processing
- Maximum 10 files per batch request
- The first request may be slower as the model loads into memory
- Embeddings are deterministic for the same input text
