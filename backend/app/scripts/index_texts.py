"""
Script to index all documents from the texts/ directory.

This script processes all text files in the texts/ directory and adds them
to the FAISS vector store for semantic search using distributed Ray processing.
"""

import os
import sys
import time
from pathlib import Path

# Get the backend directory (two levels up from this script)
backend_dir = Path(__file__).parent.parent.parent

from app.utils.document_processor import DocumentProcessor
from app.services.distributed_ingestion import DistributedIngestionPipeline
from app.services.vector_store import VectorStore


def parse_filename(filename: str) -> tuple[str, str | None]:
    """
    Parse filename to extract title and author.

    Format: Title__Author.txt (double underscore separates title from author)
    Example: Self_Reliance__Ralph_Waldo_Emerson.txt -> ("Self_Reliance", "Ralph Waldo Emerson")

    Args:
        filename: The filename to parse

    Returns:
        Tuple of (title, author) where author may be None if not present
    """
    stem = Path(filename).stem

    if "__" in stem:
        parts = stem.split("__", 1)
        title = parts[0]
        author = parts[1].replace("_", " ") if len(parts) > 1 else None
        return (title, author)

    return (stem, None)


def index_texts_directory(
    texts_dir: str = "texts",
    data_dir: str = "data",
    num_workers: int = 8,
    batch_size: int = 32,
):
    """
    Index all text files from the texts directory using distributed processing.

    Args:
        texts_dir: Directory containing text files to index
        data_dir: Directory to store the FAISS index and metadata
        num_workers: Number of Ray workers for parallel embedding
        batch_size: Batch size for embedding generation
    """
    overall_start_time = time.time()

    texts_path = Path(backend_dir) / texts_dir
    data_path = Path(backend_dir) / data_dir

    # Ensure directories exist
    if not texts_path.exists():
        print(f"Error: Texts directory not found at {texts_path}")
        return

    data_path.mkdir(exist_ok=True)

    # Clear existing index
    print("🗑️  Clearing existing FAISS index...")
    index_file = data_path / "faiss.index"
    metadata_file = data_path / "metadata.json"

    if index_file.exists():
        os.remove(index_file)
        print(f"  ✓ Removed {index_file}")
    if metadata_file.exists():
        os.remove(metadata_file)
        print(f"  ✓ Removed {metadata_file}")

    # Initialize services
    print(
        f"\n🚀 Initializing distributed ingestion pipeline ({num_workers} workers)..."
    )
    init_start = time.time()
    document_processor = DocumentProcessor()
    distributed_pipeline = DistributedIngestionPipeline(
        num_workers=num_workers, batch_size=batch_size
    )
    vector_store = VectorStore(data_dir=str(data_path))
    init_time = time.time() - init_start
    print(f"  ✓ Initialization complete in {init_time:.2f}s")

    # Find all text files
    text_files = list(texts_path.glob("*.txt"))

    if not text_files:
        print(f"No .txt files found in {texts_path}")
        distributed_pipeline.shutdown()
        return

    print(f"\n📚 Found {len(text_files)} text file(s) to index:")
    for file in text_files:
        title, author = parse_filename(file.name)
        author_str = f" by {author}" if author else ""
        print(f"  - {file.name}{author_str}")

    # Process each file
    total_sentences = 0
    total_paragraphs = 0
    total_parsing_time = 0.0
    total_embedding_time = 0.0
    total_indexing_time = 0.0
    files_processed = 0

    for text_file in text_files:
        print(f"\n{'='*70}")
        print(f"Processing: {text_file.name}")
        print(f"{'='*70}")

        try:
            # Read and parse the file
            parse_start = time.time()
            with open(text_file, "r", encoding="utf-8") as f:
                text = f.read()

            if not text.strip():
                print(f"  ⚠️  Skipping empty file: {text_file.name}")
                continue

            # Extract sentences using document processor
            sentences, paragraph_indices = (
                document_processor.split_into_sentences_and_paragraphs(text)
            )
            parse_time = time.time() - parse_start
            total_parsing_time += parse_time

            print(f"  📝 Parsed {len(sentences)} sentences in {parse_time:.2f}s")
            print(f"  📄 Detected {max(paragraph_indices) + 1} paragraphs")

            # Generate embeddings using distributed pipeline
            print(f"  ⚡ Generating embeddings with {num_workers} workers...")
            embeddings, embedding_time = (
                distributed_pipeline.embed_sentences_distributed(sentences)
            )
            total_embedding_time += embedding_time

            # Index the document
            index_start = time.time()
            print("  💾 Adding to vector store...")
            title, author = parse_filename(text_file.name)
            doc_id = vector_store.index_document(
                doc_id=f"doc_{files_processed}_{text_file.stem}",
                filename=text_file.name,
                file_type=".txt",
                sentences=sentences,
                embeddings=embeddings,
                paragraph_indices=paragraph_indices,
                author=author,
            )
            index_time = time.time() - index_start
            total_indexing_time += index_time

            print(f"  ✓ Indexed with document ID: {doc_id} ({index_time:.2f}s)")

            total_sentences += len(sentences)
            total_paragraphs += max(paragraph_indices) + 1
            files_processed += 1

        except Exception as e:
            print(f"  ❌ Error processing {text_file.name}: {str(e)}")
            import traceback

            traceback.print_exc()
            continue

    # Calculate total time
    total_time = time.time() - overall_start_time
    processing_time = total_parsing_time + total_embedding_time + total_indexing_time

    # Calculate throughput
    sentences_per_second = (
        total_sentences / total_embedding_time if total_embedding_time > 0 else 0
    )

    # Shutdown Ray
    distributed_pipeline.shutdown()

    # Print comprehensive summary
    print(f"\n{'='*70}")
    print(f"✅ INDEXING COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📊 Processing Statistics:")
    print(f"  Files processed:      {files_processed}/{len(text_files)}")
    print(f"  Total sentences:      {total_sentences:,}")
    print(f"  Total paragraphs:     {total_paragraphs:,}")
    print(
        f"  Avg sentences/file:   {total_sentences/files_processed:.1f}"
        if files_processed > 0
        else "  Avg sentences/file:   N/A"
    )

    print(f"\n⏱️  Timing Breakdown:")
    print(f"  Initialization:       {init_time:.3f}s ({init_time/total_time*100:.1f}%)")
    print(
        f"  Parsing:              {total_parsing_time:.3f}s ({total_parsing_time/total_time*100:.1f}%)"
    )
    print(
        f"  Embedding:            {total_embedding_time:.3f}s ({total_embedding_time/total_time*100:.1f}%)"
    )
    print(
        f"  Indexing:             {total_indexing_time:.3f}s ({total_indexing_time/total_time*100:.1f}%)"
    )
    print(f"  Other overhead:       {total_time - init_time - processing_time:.3f}s")
    print(f"  ─────────────────────────────────")
    print(f"  TOTAL TIME:           {total_time:.3f}s")

    print(f"\n🚀 Performance Metrics:")
    print(f"  Embedding throughput: {sentences_per_second:.1f} sentences/sec")
    print(
        f"  Processing speed:     {total_sentences/processing_time:.1f} sentences/sec"
        if processing_time > 0
        else "  Processing speed:     N/A"
    )
    print(f"  Workers used:         {num_workers}")

    print(f"\n💾 Output Files:")
    print(f"  Index:                {data_path / 'faiss.index'}")
    print(f"  Metadata:             {data_path / 'metadata.json'}")

    # Print indexed documents
    docs = vector_store.list_documents()
    print(f"\n📚 Indexed Documents ({len(docs)}):")
    for doc in docs:
        print(
            f"  - {doc.filename:40} {doc.total_sentences:>5} sentences, {doc.total_paragraphs:>4} paragraphs"
        )

    print(f"\n{'='*70}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Index all text files from the texts/ directory using distributed Ray processing"
    )
    parser.add_argument(
        "--texts-dir",
        default="texts",
        help="Directory containing text files (default: texts)",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory to store index and metadata (default: data)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of Ray workers for parallel embedding (default: 8)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation (default: 32)",
    )

    args = parser.parse_args()

    index_texts_directory(args.texts_dir, args.data_dir, args.workers, args.batch_size)
