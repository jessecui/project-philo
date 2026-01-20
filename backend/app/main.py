from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException, Cookie, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import uuid
import json
from pathlib import Path
from typing import Optional, AsyncGenerator
from pydantic import BaseModel
from app.services.embedding_service import EmbeddingService
from app.utils.document_processor import DocumentProcessor
from app.services.vector_store import VectorStore
from app.services.distributed_ingestion import DistributedIngestionPipeline
from app.services.reranker_service import CrossEncoderReranker
from app.services.generation_service import GeminiGenerator
import time

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

app = FastAPI(
    title="Document Embedding API",
    docs_url=None,  # Disable /docs
    redoc_url=None,  # Disable /redoc
    openapi_url=None,  # Disable /openapi.json
)

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
reranker = CrossEncoderReranker()  # Initialize reranker for 2-stage retrieval


# --- Auth Dependency ---


async def require_auth(creator_auth: str | None = Cookie(default=None)):
    """Require valid auth cookie to access protected endpoints."""
    if creator_auth != "true":
        raise HTTPException(status_code=401, detail="Unauthorized")


# Initialize distributed ingestion pipeline only if ENABLE_RAY is set
if os.getenv("ENABLE_RAY", "false").lower() == "true":
    distributed_pipeline = DistributedIngestionPipeline(num_workers=8, batch_size=32)
else:
    print("ℹ️  Ray distributed processing disabled. Set ENABLE_RAY=true to enable.")
    distributed_pipeline = None

# Initialize Gemini generator (only if API key is configured)
try:
    generator = GeminiGenerator()
except Exception as e:
    print(f"⚠️  Gemini generator not initialized: {e}")
    print("   RAG generation endpoints will not be available.")
    print("   Set GOOGLE_API_KEY in .env to enable generation features.")
    generator = None

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# Request models
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    deduplicate_paragraphs: bool = True
    use_reranking: bool = False  # Enable 2-stage retrieval with cross-encoder reranking
    top_k_faiss: int = 30  # Number of candidates from FAISS (stage 1)
    context_window: int = 2  # Number of paragraphs before/after for context expansion


class GenerateRequest(BaseModel):
    query: str
    top_k_context: int = 5  # Number of document excerpts to retrieve
    use_reranking: bool = True  # Use cross-encoder reranking for better quality
    top_k_faiss: int = 30  # FAISS candidates (if reranking enabled)
    context_window: int = 0  # Paragraphs before/after (0 = main paragraph only)
    temperature: float = 0.7  # Sampling temperature for generation
    max_output_tokens: int = 8192  # Maximum tokens in generated response


class GenerateFromResultsRequest(BaseModel):
    query: str
    results: list  # Pre-fetched search results from /search endpoint
    temperature: float = 0.7  # Sampling temperature for generation
    max_output_tokens: int = 8192  # Maximum tokens in generated response


@app.get("/health")
async def health():
    return {"status": "healthy"}


class ValidateCreatorRequest(BaseModel):
    answer: str


@app.post("/validate-creator")
async def validate_creator(request: ValidateCreatorRequest):
    """
    Validate the creator's name against the stored environment variable.

    Args:
        answer: The user's answer to the creator question

    Returns:
        {"valid": true/false}
    """
    creator_name = os.getenv("CREATOR_NAME", "").strip().replace(" ", "").lower()
    user_answer = request.answer.strip().replace(" ", "").lower()

    if not creator_name:
        raise HTTPException(
            status_code=500, detail="CREATOR_NAME environment variable not configured"
        )

    return {"valid": user_answer == creator_name}


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


@app.post("/index-distributed")
async def index_document_distributed(file: UploadFile = File(...)):
    """
    Upload and index a document using distributed Ray workers for faster processing.
    Ideal for large documents (1000+ sentences).

    Returns:
        - doc_id: Unique document identifier
        - filename: Original filename
        - file_type: Type of file uploaded
        - sentence_count: Number of sentences indexed
        - paragraph_count: Number of paragraphs detected
        - processing_time: Total time in seconds
        - embedding_time: Time spent on embedding generation
        - throughput: Sentences per second
        - workers: Number of Ray workers used
    """
    # Check if Ray is enabled
    if distributed_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Distributed processing not available. Set ENABLE_RAY=true to enable.",
        )

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
        start_time = time.time()

        # Save uploaded file temporarily
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Extract text from document
        parse_start = time.time()
        text = document_processor.extract_text(file_path, file_ext)
        parse_time = time.time() - parse_start

        if not text.strip():
            raise HTTPException(
                status_code=400, detail="No text could be extracted from the document"
            )

        # Split into sentences and paragraphs
        split_start = time.time()
        sentences, paragraph_indices = (
            document_processor.split_into_sentences_and_paragraphs(text)
        )
        split_time = time.time() - split_start

        # Generate embeddings (distributed)
        embeddings, metrics = distributed_pipeline.process_document(
            sentences, paragraph_indices
        )

        # Clean up uploaded file
        os.remove(file_path)

        # Generate unique document ID
        doc_id = str(uuid.uuid4())

        # Index in vector store
        index_start = time.time()
        success = vector_store.index_document(
            doc_id=doc_id,
            filename=file.filename,
            file_type=file_ext,
            sentences=sentences,
            paragraph_indices=paragraph_indices,
            embeddings=embeddings,
        )
        index_time = time.time() - index_start

        if not success:
            raise HTTPException(status_code=500, detail="Failed to index document")

        total_time = time.time() - start_time
        paragraph_count = max(paragraph_indices) + 1 if paragraph_indices else 0

        return JSONResponse(
            content={
                "doc_id": doc_id,
                "filename": file.filename,
                "file_type": file_ext,
                "sentence_count": len(sentences),
                "paragraph_count": paragraph_count,
                "message": "Document indexed successfully with distributed processing",
                "performance": {
                    "total_time": round(total_time, 2),
                    "parsing_time": round(parse_time, 2),
                    "splitting_time": round(split_time, 2),
                    "embedding_time": round(metrics.embedding_time, 2),
                    "indexing_time": round(index_time, 2),
                    "throughput": round(metrics.sentences_per_second, 1),
                    "workers": distributed_pipeline.num_workers,
                },
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
        top_k: Number of final results to return (default: 5)
        deduplicate_paragraphs: Return only one result per paragraph (default: True)
        use_reranking: Enable 2-stage retrieval with cross-encoder reranking (default: False)
        top_k_faiss: Number of candidates from FAISS stage (default: 50, only used if use_reranking=True)
        context_window: Number of paragraphs before/after for context expansion (default: 2)

    Returns:
        List of search results with matched sentences and paragraph context
        If use_reranking=True, also includes reranking_score and timing breakdown
    """
    try:
        # Generate query embedding
        query_embedding = embedding_service.embed_texts([request.query])[0]

        # Choose retrieval method
        if request.use_reranking:
            # 2-stage retrieval: FAISS + cross-encoder reranking
            results, timing = vector_store.search_with_reranking(
                query_text=request.query,
                query_embedding=query_embedding,
                reranker=reranker,
                top_k_faiss=request.top_k_faiss,
                top_k_paragraphs=request.top_k,
                context_window=request.context_window,
            )

            # Convert results to JSON-serializable format (with reranking data)
            response_results = []
            for result in results:
                result_dict = {
                    "doc_id": result.doc_id,
                    "filename": result.filename,
                    "author": result.author,
                    "paragraph_idx": result.paragraph_idx,
                    "paragraph_text": result.paragraph_text,
                    "reranking_score": (
                        float(result.reranking_score)
                        if result.reranking_score is not None
                        else None
                    ),
                }

                # Add context paragraphs if present
                if result.context_paragraphs_before:
                    result_dict["context_paragraphs_before"] = (
                        result.context_paragraphs_before
                    )

                if result.context_paragraphs_after:
                    result_dict["context_paragraphs_after"] = (
                        result.context_paragraphs_after
                    )

                response_results.append(result_dict)

            return JSONResponse(
                content={
                    "query": request.query,
                    "results": response_results,
                    "total_results": len(response_results),
                    "timing": {
                        "faiss_time": round(timing["faiss_time"], 3),
                        "reranking_time": round(timing["reranking_time"], 3),
                        "total_time": round(timing["total_time"], 3),
                    },
                    "retrieval_method": "faiss_with_reranking",
                }
            )

        else:
            # FAISS-only retrieval
            results = vector_store.search(
                query_embedding=query_embedding,
                top_k=request.top_k,
            )

            # Convert results to JSON-serializable format
            response_results = []
            for result in results:
                response_results.append(
                    {
                        "doc_id": result.doc_id,
                        "filename": result.filename,
                        "author": result.author,
                        "paragraph_idx": result.paragraph_idx,
                        "paragraph_text": result.paragraph_text,
                    }
                )

            return JSONResponse(
                content={
                    "query": request.query,
                    "results": response_results,
                    "total_results": len(response_results),
                    "retrieval_method": "faiss_only",
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


@app.post("/generate")
async def generate_from_results(request: GenerateFromResultsRequest):
    """
    Generate an answer from pre-fetched search results.

    This endpoint enables a two-step RAG workflow:
    1. User calls /search to get and review results
    2. User clicks a button to generate an answer from those results

    Args:
        query: User's question
        results: List of search results from /search endpoint
        temperature: Generation temperature (default: 0.7)
        max_output_tokens: Maximum response length (default: 8192)

    Returns:
        Server-Sent Events stream with:
        - sources: Initial event with the provided search results
        - token: Streaming text chunks from the model
        - done: Final event with timing information
    """
    if generator is None:
        raise HTTPException(
            status_code=503,
            detail="RAG generation not available. Vertex AI is not configured. "
            "Please set up .env file with GOOGLE_CLOUD_PROJECT and credentials.",
        )

    try:
        if not request.results:
            raise HTTPException(
                status_code=400,
                detail="No search results provided. Please provide results from /search endpoint.",
            )

        overall_start = time.time()

        # Convert the JSON results back to SearchResult objects
        from app.services.vector_store import SearchResult

        search_results = []
        for r in request.results:
            result = SearchResult(
                doc_id=r.get("doc_id"),
                filename=r.get("filename"),
                paragraph_idx=r.get("paragraph_idx"),
                paragraph_text=r.get("paragraph_text"),
                reranking_score=r.get("reranking_score"),
                context_paragraphs_before=r.get("context_paragraphs_before"),
                context_paragraphs_after=r.get("context_paragraphs_after"),
            )
            search_results.append(result)

        # Stream generated answer
        async def event_generator() -> AsyncGenerator[str, None]:
            """Generate Server-Sent Events for streaming response."""
            try:
                # Send initial sources event
                sources_data = [
                    {
                        "filename": r.filename,
                        "author": r.author,
                        "paragraph_idx": r.paragraph_idx,
                        "text": r.paragraph_text,
                        "score": (
                            float(r.reranking_score)
                            if r.reranking_score is not None
                            else None
                        ),
                    }
                    for r in search_results
                ]
                yield f"data: {json.dumps({'type': 'sources', 'data': sources_data})}\n\n"

                # Stream answer tokens
                generation_start = time.time()
                async for chunk in generator.stream_answer(
                    query=request.query,
                    search_results=search_results,
                    temperature=request.temperature,
                    max_output_tokens=request.max_output_tokens,
                ):
                    yield f"data: {json.dumps({'type': 'token', 'data': chunk})}\n\n"

                generation_time = time.time() - generation_start
                total_time = time.time() - overall_start

                # Send completion event with timing
                timing_data = {
                    "generation_time": round(generation_time, 3),
                    "total_time": round(total_time, 3),
                }
                yield f"data: {json.dumps({'type': 'done', 'data': timing_data})}\n\n"

            except Exception as e:
                error_data = {"message": str(e), "code": "generation_error"}
                yield f"data: {json.dumps({'type': 'error', 'data': error_data})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search-and-generate", dependencies=[Depends(require_auth)])
async def search_and_generate(request: GenerateRequest):
    """
    Retrieve relevant document excerpts and generate an answer using Gemini 3 Flash.

    This endpoint implements RAG (Retrieval-Augmented Generation):
    1. Embeds the query
    2. Retrieves top-k relevant passages with optional reranking
    3. Streams an AI-generated answer grounded in the retrieved context

    Args:
        query: User's question
        top_k_context: Number of document excerpts to retrieve (default: 5)
        use_reranking: Enable cross-encoder reranking (default: True)
        top_k_faiss: FAISS candidates for reranking (default: 50)
        context_window: Paragraphs before/after for context (default: 2)
        temperature: Generation temperature (default: 0.7)
        max_output_tokens: Maximum response length (default: 2048)

    Returns:
        Server-Sent Events stream with:
        - sources: Initial event with retrieved document excerpts
        - token: Streaming text chunks from the model
        - done: Final event with timing information
    """
    if generator is None:
        raise HTTPException(
            status_code=503,
            detail="RAG generation not available. Vertex AI is not configured. "
            "Please set up .env file with GOOGLE_CLOUD_PROJECT and credentials.",
        )

    try:
        overall_start = time.time()

        # Step 1: Generate query embedding
        embed_start = time.time()
        query_embedding = embedding_service.embed_texts([request.query])[0]
        embed_time = time.time() - embed_start

        # Step 2: Retrieve relevant passages
        retrieval_start = time.time()
        if request.use_reranking:
            results, timing = vector_store.search_with_reranking(
                query_text=request.query,
                query_embedding=query_embedding,
                reranker=reranker,
                top_k_faiss=request.top_k_faiss,
                top_k_paragraphs=request.top_k_context,
                context_window=request.context_window,
            )
            retrieval_time = timing["total_time"]
        else:
            results = vector_store.search(
                query_embedding=query_embedding,
                top_k=request.top_k_context,
            )
            retrieval_time = time.time() - retrieval_start
            timing = {"faiss_time": retrieval_time, "reranking_time": 0}

        if not results:
            raise HTTPException(
                status_code=404,
                detail="No relevant documents found. Please index documents first.",
            )

        # Step 3: Stream generated answer
        async def event_generator() -> AsyncGenerator[str, None]:
            """Generate Server-Sent Events for streaming response."""
            try:
                # Send initial sources event
                sources_data = [
                    {
                        "filename": r.filename,
                        "author": r.author,
                        "paragraph_idx": r.paragraph_idx,
                        "text": r.paragraph_text,
                        "score": (
                            float(r.reranking_score)
                            if r.reranking_score is not None
                            else None
                        ),
                    }
                    for r in results
                ]
                yield f"data: {json.dumps({'type': 'sources', 'data': sources_data})}\n\n"

                # Stream answer tokens
                generation_start = time.time()
                async for chunk in generator.stream_answer(
                    query=request.query,
                    search_results=results,
                    temperature=request.temperature,
                    max_output_tokens=request.max_output_tokens,
                ):
                    yield f"data: {json.dumps({'type': 'token', 'data': chunk})}\n\n"

                generation_time = time.time() - generation_start
                total_time = time.time() - overall_start

                # Send completion event with timing
                timing_data = {
                    "embedding_time": round(embed_time, 3),
                    "faiss_time": round(timing["faiss_time"], 3),
                    "reranking_time": round(timing["reranking_time"], 3),
                    "retrieval_time": round(retrieval_time, 3),
                    "generation_time": round(generation_time, 3),
                    "total_time": round(total_time, 3),
                }
                yield f"data: {json.dumps({'type': 'done', 'data': timing_data})}\n\n"

            except Exception as e:
                error_data = {"message": str(e), "code": "generation_error"}
                yield f"data: {json.dumps({'type': 'error', 'data': error_data})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Serve React Frontend ---

# Mount static assets if frontend is built
if FRONTEND_DIST.exists() and (FRONTEND_DIST / "assets").exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(FRONTEND_DIST / "assets")),
        name="frontend-assets",
    )


@app.get("/", summary="Health check / Frontend", include_in_schema=False)
async def root():
    """Serve frontend index.html in production, health check in dev."""
    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "status": "healthy",
        "message": "API running. Build frontend with: cd frontend && npm run build",
    }


@app.get("/{path:path}", include_in_schema=False)
async def spa_fallback(path: str):
    """Serve index.html for any unmatched routes (SPA client-side routing)."""
    # First check if it's a static file in dist root (favicon.svg, etc.)
    file_path = FRONTEND_DIST / path
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
    # Otherwise serve index.html for client-side routing
    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="Not found")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
