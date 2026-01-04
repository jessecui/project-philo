"""
Test script for FAISS retrieval + cross-encoder reranking.

This script demonstrates the two-stage retrieval process:
1. FAISS vector similarity search (fast candidate retrieval)
2. Cross-encoder reranking (precise relevance scoring)
"""

import sys
from pathlib import Path

# Get the backend directory
backend_dir = Path(__file__).parent.parent.parent

from app.services.vector_store import VectorStore
from app.services.embedding_service import EmbeddingService
from app.services.reranker_service import CrossEncoderReranker


def test_search(query: str, top_k_faiss: int = 50, top_n_final: int = 10):
    """
    Test search with FAISS + cross-encoder reranking.

    Args:
        query: Search query text
        top_k_faiss: Number of candidates from FAISS (default: 50)
        top_n_final: Number of final results after reranking (default: 10)
    """
    print(f"\n{'='*80}")
    print(f"Testing Search Pipeline")
    print(f"{'='*80}")
    print(f'Query: "{query}"')
    print(f"FAISS candidates: {top_k_faiss}")
    print(f"Final results: {top_n_final}")

    # Initialize services
    print(f"\n{'='*80}")
    print("Initializing services...")
    print(f"{'='*80}")

    data_dir = str(backend_dir / "data")

    vector_store = VectorStore(data_dir=data_dir)
    embedding_service = EmbeddingService()
    reranker = CrossEncoderReranker()

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

    # Generate query embedding
    print(f"\n{'='*80}")
    print("Stage 1: Generating query embedding...")
    print(f"{'='*80}")
    query_embedding = embedding_service.embed_texts([query])[0]
    print(f"✓ Query embedding generated (dimension: {len(query_embedding)})")

    # Perform search with reranking
    print(f"\n{'='*80}")
    print("Stage 2: FAISS retrieval + Cross-encoder reranking...")
    print(f"{'='*80}")

    results, timing_info = vector_store.search_with_reranking(
        query_text=query,
        query_embedding=query_embedding,
        reranker=reranker,
        top_k_faiss=top_k_faiss,
        top_n_final=top_n_final,
        context_window=2,  # Include 2 paragraphs before/after for context
    )

    # Print timing information
    print(f"\n⏱️  Timing Breakdown:")
    print(f"  FAISS retrieval:      {timing_info['faiss_time']:.3f}s")
    print(f"  Cross-encoder:        {timing_info['reranking_time']:.3f}s")
    print(f"  ─────────────────────────")
    print(f"  Total time:           {timing_info['total_time']:.3f}s")

    # Print results
    print(f"\n{'='*80}")
    print(f"Search Results ({len(results)} results)")
    print(f"{'='*80}")

    if not results:
        print("\nNo results found.")
        return

    for i, result in enumerate(results, 1):
        print(f"\n{'─'*80}")
        print(f"Result #{i}")
        print(f"{'─'*80}")
        print(f"📄 Document:          {result.filename}")
        print(f"📍 Paragraph:         {result.paragraph_idx}")
        print(f"🎯 Reranking Score:   {result.reranking_score:.4f}")
        print(
            f"💬 FAISS Scores:      {[f'{s:.4f}' for s in result.similarity_scores[:3]]}"
        )

        # Show context paragraphs before (if available)
        if result.context_paragraphs_before:
            print(f"\n📖 Context (before):")
            for ctx_para in result.context_paragraphs_before:
                print(
                    f"   {ctx_para[:150]}..."
                    if len(ctx_para) > 150
                    else f"   {ctx_para}"
                )

        # Show matched paragraph
        print(f"\n✨ Matched Paragraph:")
        print(f"   {result.paragraph_text}")

        # Show matched sentences
        if result.matched_sentences:
            print(f"\n🔍 Matched Sentences ({len(result.matched_sentences)}):")
            for j, sentence in enumerate(result.matched_sentences[:3], 1):
                print(f"   {j}. {sentence}")

        # Show context paragraphs after (if available)
        if result.context_paragraphs_after:
            print(f"\n📖 Context (after):")
            for ctx_para in result.context_paragraphs_after:
                print(
                    f"   {ctx_para[:150]}..."
                    if len(ctx_para) > 150
                    else f"   {ctx_para}"
                )

    print(f"\n{'='*80}")
    print("Search complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test FAISS + cross-encoder search pipeline"
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
        "--top-n",
        type=int,
        default=10,
        help="Number of final results after reranking (default: 10)",
    )

    args = parser.parse_args()

    test_search(query=args.query, top_k_faiss=args.top_k_faiss, top_n_final=args.top_n)
