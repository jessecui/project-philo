from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import uuid
from typing import Optional
from pydantic import BaseModel
from app.services.embedding_service import EmbeddingService
from app.utils.document_processor import DocumentProcessor
from app.services.vector_store import VectorStore

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
vector_store = VectorStore()

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# Request models
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    deduplicate_paragraphs: bool = True


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


@app.post("/index")
async def index_document(file: UploadFile = File(...)):
    """
    Upload and index a document for semantic search.

    Returns:
        - doc_id: Unique document identifier
        - filename: Original filename
        - file_type: Type of file uploaded
        - sentence_count: Number of sentences indexed
        - paragraph_count: Number of paragraphs detected
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

        # Generate unique document ID
        doc_id = str(uuid.uuid4())

        # Index in vector store
        success = vector_store.index_document(
            doc_id=doc_id,
            filename=file.filename,
            file_type=file_ext,
            sentences=sentences,
            paragraph_indices=paragraph_indices,
            embeddings=embeddings,
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to index document")

        paragraph_count = max(paragraph_indices) + 1 if paragraph_indices else 0

        return JSONResponse(
            content={
                "doc_id": doc_id,
                "filename": file.filename,
                "file_type": file_ext,
                "sentence_count": len(sentences),
                "paragraph_count": paragraph_count,
                "message": "Document indexed successfully",
            }
        )

    except Exception as e:
        # Clean up file if it exists
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search_documents(request: SearchRequest):
    """
    Search indexed documents for semantically similar passages.

    Args:
        query: Search query text
        top_k: Number of results to return (default: 5)
        deduplicate_paragraphs: Return only one result per paragraph (default: True)

    Returns:
        List of search results with matched sentences and paragraph context
    """
    try:
        # Generate query embedding
        query_embedding = embedding_service.embed_texts([request.query])[0]

        # Search vector store
        results = vector_store.search(
            query_embedding=query_embedding,
            top_k=request.top_k,
            deduplicate_paragraphs=request.deduplicate_paragraphs,
        )

        # Convert results to JSON-serializable format
        response_results = []
        for result in results:
            response_results.append(
                {
                    "doc_id": result.doc_id,
                    "filename": result.filename,
                    "paragraph_index": result.paragraph_idx,
                    "paragraph_text": result.paragraph_text,
                    "matched_sentences": result.matched_sentences,
                    "similarity_scores": [
                        float(score) for score in result.similarity_scores
                    ],
                }
            )

        return JSONResponse(
            content={
                "query": request.query,
                "results": response_results,
                "total_results": len(response_results),
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents():
    """
    List all indexed documents.

    Returns:
        List of document metadata
    """
    try:
        documents = vector_store.list_documents()

        return JSONResponse(
            content={
                "documents": [
                    {
                        "doc_id": doc.doc_id,
                        "filename": doc.filename,
                        "file_type": doc.file_type,
                        "total_sentences": doc.total_sentences,
                        "total_paragraphs": doc.total_paragraphs,
                    }
                    for doc in documents
                ],
                "total_documents": len(documents),
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """
    Delete a document from the index.

    Args:
        doc_id: Document ID to delete

    Returns:
        Success message
    """
    try:
        success = vector_store.delete_document(doc_id)

        if not success:
            raise HTTPException(
                status_code=404, detail=f"Document with ID {doc_id} not found"
            )

        return JSONResponse(
            content={"message": "Document deleted successfully", "doc_id": doc_id}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """
    Get statistics about the vector store.

    Returns:
        Statistics including total documents, sentences, and paragraphs
    """
    try:
        stats = vector_store.get_stats()
        return JSONResponse(content=stats)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
