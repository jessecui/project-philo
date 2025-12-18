from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from typing import List
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
    Upload a document (PDF, TXT, MD, or DOCX) and get its embeddings.
    
    Returns:
        - filename: Original filename
        - file_type: Type of file uploaded
        - text_length: Number of characters in extracted text
        - embedding_dimension: Dimension of the embedding vector
        - embedding: The embedding vector as a list
    """
    # Validate file type
    allowed_extensions = {".pdf", ".txt", ".md", ".docx"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
        )
    
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
                status_code=400,
                detail="No text could be extracted from the document"
            )
        
        # Generate embedding
        embedding = embedding_service.embed_text(text)
        
        # Clean up uploaded file
        os.remove(file_path)
        
        return JSONResponse(content={
            "filename": file.filename,
            "file_type": file_ext,
            "text_length": len(text),
            "text_preview": text[:200] + "..." if len(text) > 200 else text,
            "embedding_dimension": len(embedding),
            "embedding": embedding
        })
        
    except Exception as e:
        # Clean up file if it exists
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed-batch")
async def embed_documents_batch(files: List[UploadFile] = File(...)):
    """
    Upload multiple documents and get their embeddings.
    
    Returns a list of embedding results for each document.
    """
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 files allowed per batch"
        )
    
    results = []
    
    for file in files:
        # Validate file type
        allowed_extensions = {".pdf", ".txt", ".md", ".docx"}
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            results.append({
                "filename": file.filename,
                "error": f"Unsupported file type: {file_ext}"
            })
            continue
        
        try:
            # Save uploaded file temporarily
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Extract text from document
            text = document_processor.extract_text(file_path, file_ext)
            
            if not text.strip():
                results.append({
                    "filename": file.filename,
                    "error": "No text could be extracted"
                })
                os.remove(file_path)
                continue
            
            # Generate embedding
            embedding = embedding_service.embed_text(text)
            
            # Clean up uploaded file
            os.remove(file_path)
            
            results.append({
                "filename": file.filename,
                "file_type": file_ext,
                "text_length": len(text),
                "text_preview": text[:200] + "..." if len(text) > 200 else text,
                "embedding_dimension": len(embedding),
                "embedding": embedding
            })
            
        except Exception as e:
            # Clean up file if it exists
            if os.path.exists(file_path):
                os.remove(file_path)
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return JSONResponse(content={"results": results})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
