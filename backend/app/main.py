from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from app.services.embedding_service import EmbeddingService
from app.utils.document_processor import DocumentProcessor

app = FastAPI(title="Document Embedding API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
embedding_service = EmbeddingService()
document_processor = DocumentProcessor()

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
async def root():
    return {"message": "Document Embedding API", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/embed")
async def embed_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF, TXT, MD, or DOCX) and get sentence-level embeddings.

    Returns:
        - filename: Original filename
        - file_type: Type of file uploaded
        - text_length: Number of characters in extracted text
        - sentence_count: Number of sentences extracted
        - paragraph_count: Number of paragraphs detected
        - sentences: List of sentence strings
        - paragraph_indices: List of paragraph indices for each sentence
        - embeddings: List of embedding vectors for each sentence
        - embedding_dimension: Dimension of each embedding vector
    """
    # Validate file type
    allowed_extensions = {".pdf", ".txt", ".md", ".docx"}
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}",
        )

    file_path = None
    try:
        # Save uploaded file temporarily
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Extract text from document
        text = document_processor.extract_text(file_path, file_ext)

        if not text.strip():
            raise HTTPException(
                status_code=400, detail="No text could be extracted from the document"
            )

        # Generate sentence-level embeddings
        sentences, paragraph_indices, embeddings = embedding_service.embed_by_sentence(
            text, document_processor
        )

        # Clean up uploaded file
        os.remove(file_path)

        # Calculate paragraph count (max index + 1)
        paragraph_count = max(paragraph_indices) + 1 if paragraph_indices else 0

        return JSONResponse(
            content={
                "filename": file.filename,
                "file_type": file_ext,
                "text_length": len(text),
                "sentence_count": len(sentences),
                "paragraph_count": paragraph_count,
                "sentences": sentences,
                "paragraph_indices": paragraph_indices,
                "embeddings": embeddings,
                "embedding_dimension": len(embeddings[0]) if embeddings else 0,
            }
        )

    except Exception as e:
        # Clean up file if it exists
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
