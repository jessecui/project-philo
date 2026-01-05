"""
Test script for RAG (Retrieval-Augmented Generation) pipeline.

This script demonstrates the full RAG workflow:
1. FAISS vector similarity search (fast candidate retrieval)
2. Cross-encoder reranking (precise relevance scoring)
3. Gemini 2.5 Pro generation (grounded answer with citations)
"""

import sys
from pathlib import Path
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Get the backend directory
backend_dir = Path(__file__).parent.parent.parent

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv(backend_dir / ".env")

from app.services.vector_store import VectorStore
from app.services.embedding_service import EmbeddingService
from app.services.reranker_service import CrossEncoderReranker
from app.services.generation_service import VertexAIGenerator
import asyncio


async def test_search_and_generate(
    query: str,
    top_k_faiss: int = 50,
    top_n_context: int = 5,
    temperature: float = 0.7,
    max_output_tokens: int = 2048,
):
    """
    Test RAG pipeline with streaming generation.

    Args:
        query: Search query text
        top_k_faiss: Number of candidates from FAISS (default: 50)
        top_n_context: Number of excerpts to use for generation (default: 5)
        temperature: Generation temperature (default: 0.7)
        max_output_tokens: Maximum tokens in response (default: 2048)
    """
    print(f"\n{'='*80}")
    print(f"Testing RAG Pipeline (Retrieval + Generation)")
    print(f"{'='*80}")
    print(f'Query: "{query}"')
    print(f"FAISS candidates: {top_k_faiss}")
    print(f"Context excerpts: {top_n_context}")
    print(f"Temperature: {temperature}")

    # Initialize services
    print(f"\n{'='*80}")
    print("Initializing services...")
    print(f"{'='*80}")

    data_dir = str(backend_dir / "data")

    vector_store = VectorStore(data_dir=data_dir)
    embedding_service = EmbeddingService()
    reranker = CrossEncoderReranker()

    # Initialize generator (may fail if Vertex AI not configured)
    try:
        generator = VertexAIGenerator(
            temperature=temperature, max_output_tokens=max_output_tokens
        )
    except Exception as e:
        print(f"\n❌ Error initializing Vertex AI generator: {e}")
        print("\nTo use generation features:")
        print("1. Set up .env file with Google Cloud credentials")
        print("2. See backend/.env.example for configuration")
        return

    # Check if index has data
    if vector_store.index.ntotal == 0:
        print("\n❌ Error: No documents in the index!")
        print("Run: python -m app.scripts.index_texts")
        return

    print(f"\n✓ Vector store loaded: {vector_store.index.ntotal} sentences indexed")

    # List indexed documents
    docs = vector_store.list_documents()
    print(f"\n📚 Indexed documents ({len(docs)}):")
    for doc in docs:
        print(
            f"  - {doc.filename}: {doc.total_sentences} sentences, {doc.total_paragraphs} paragraphs"
        )

    # ========================================================================
    # STAGE 1: Retrieval
    # ========================================================================
    print(f"\n{'='*80}")
    print("STAGE 1: Document Retrieval")
    print(f"{'='*80}")

    # Generate query embedding
    print(f"\n  → Generating query embedding...")
    query_embedding = embedding_service.embed_texts([query])[0]
    print(f"  ✓ Query embedding generated (dimension: {len(query_embedding)})")

    # Perform search with reranking
    print(f"\n  → FAISS retrieval + Cross-encoder reranking...")
    results, timing_info = vector_store.search_with_reranking(
        query_text=query,
        query_embedding=query_embedding,
        reranker=reranker,
        top_k_faiss=top_k_faiss,
        top_n_final=top_n_context,
        context_window=2,  # Include 2 paragraphs before/after for context
    )

    # Print timing information
    print(f"\n  ⏱️  Retrieval Timing:")
    print(f"     FAISS retrieval:      {timing_info['faiss_time']:.3f}s")
    print(f"     Cross-encoder:        {timing_info['reranking_time']:.3f}s")
    print(f"     ─────────────────────────")
    print(f"     Total retrieval:      {timing_info['total_time']:.3f}s")

    # Print retrieved excerpts
    print(f"\n  📝 Retrieved Context ({len(results)} excerpts):")
    for i, result in enumerate(results, 1):
        print(f"\n     [{i}] {result.filename} (¶{result.paragraph_idx})")
        print(f"         Relevance: {result.reranking_score:.4f}")
        preview = result.paragraph_text[:100].replace("\n", " ")
        print(f"         Preview: {preview}...")

    if not results:
        print("\n❌ No results found. Cannot generate answer.")
        return

    # ========================================================================
    # STAGE 2: Generation
    # ========================================================================
    print(f"\n{'='*80}")
    print("STAGE 2: Answer Generation (Streaming)")
    print(f"{'='*80}")
    print(f"\n💬 Generated Answer:\n")

    # Stream the answer
    import time

    generation_start = time.time()
    token_count = 0

    try:
        async for chunk in generator.stream_answer(
            query=query,
            search_results=results,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ):
            print(chunk, end="", flush=True)
            token_count += 1

        generation_time = time.time() - generation_start

        # Print generation statistics
        print(f"\n\n{'='*80}")
        print("Generation Complete")
        print(f"{'='*80}")
        print(f"\n⏱️  Generation Timing:")
        print(f"   Generation time:      {generation_time:.3f}s")
        print(f"   Chunks streamed:      {token_count}")
        print(f"   ─────────────────────────")
        print(
            f"   Total pipeline:       {timing_info['total_time'] + generation_time:.3f}s"
        )

    except Exception as e:
        print(f"\n\n❌ Error during generation: {e}")
        import traceback

        traceback.print_exc()

    print(f"\n{'='*80}")
    print("RAG Pipeline Complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test RAG pipeline with FAISS + cross-encoder + Gemini generation"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="What is the nature of the Tao?",
        help="Search query (default: 'What is the nature of the Tao?')",
    )
    parser.add_argument(
        "--top-k-faiss",
        type=int,
        default=50,
        help="Number of candidates from FAISS (default: 50)",
    )
    parser.add_argument(
        "--top-n-context",
        type=int,
        default=5,
        help="Number of context excerpts for generation (default: 5)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature 0.0-1.0 (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum output tokens (default: 2048)",
    )

    args = parser.parse_args()

    # Run async function
    asyncio.run(
        test_search_and_generate(
            query=args.query,
            top_k_faiss=args.top_k_faiss,
            top_n_context=args.top_n_context,
            temperature=args.temperature,
            max_output_tokens=args.max_tokens,
        )
    )
