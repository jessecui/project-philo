"""
Benchmarking utilities for comparing single core (non-Ray) vs 10 cores with Ray.
"""

import time
import os
import random
import string
import shutil
import torch
from typing import Dict, Any
from app.services.embedding_service import EmbeddingService
from app.services.distributed_ingestion import DistributedIngestionPipeline
from app.services.vector_store import VectorStore
from app.utils.document_processor import DocumentProcessor


def clear_faiss_data(data_dir: str = "data"):
    """Clear existing FAISS data directory."""
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir, exist_ok=True)


def benchmark_ingestion(text: str, batch_size: int = 32) -> Dict[str, Any]:
    """
    Benchmark single core CPU (non-Ray) vs 10 cores with Ray (CPU only).

    Args:
        text: Input text to process
        batch_size: Batch size per worker

    Returns:
        Dictionary with benchmark results
    """
    # Force PyTorch to use only 1 thread for true single-core comparison
    torch.set_num_threads(1)
    print(
        f"Set PyTorch to use {torch.get_num_threads()} thread(s) for sequential benchmark\n"
    )

    # Initialize services
    doc_processor = DocumentProcessor()
    embedding_service = EmbeddingService(force_cpu=True)

    # Process text
    print("Processing text...")
    sentences, paragraph_indices = doc_processor.split_into_sentences_and_paragraphs(
        text
    )

    # Add unique random word to each sentence to prevent duplicate embeddings
    print("Adding unique tokens to sentences to prevent caching...")
    unique_sentences = []
    for sentence in sentences:
        random_word = "".join(random.choices(string.ascii_lowercase, k=8))
        unique_sentences.append(f"{sentence.rstrip('.')} [{random_word}].")

    sentences = unique_sentences

    print(
        f"Extracted {len(sentences)} sentences across {max(paragraph_indices) + 1 if paragraph_indices else 0} paragraphs"
    )

    # Single core CPU non-Ray benchmark
    print("\n=== Single Core CPU Non-Ray (Sequential) Ingestion ===")
    print(f"Device: {embedding_service.device}")

    # Clear FAISS before sequential test
    print("Clearing FAISS data...")
    clear_faiss_data()
    vector_store_seq = VectorStore()

    seq_start = time.time()
    seq_embeddings = embedding_service.embed_texts(sentences)
    seq_embed_time = time.time() - seq_start

    # Index to FAISS
    faiss_start = time.time()
    vector_store_seq.index_document(
        doc_id="benchmark_seq",
        filename="benchmark_sequential.txt",
        file_type="txt",
        sentences=sentences,
        paragraph_indices=paragraph_indices,
        embeddings=seq_embeddings,
    )
    seq_faiss_time = time.time() - faiss_start
    seq_total_time = seq_embed_time + seq_faiss_time
    seq_throughput = len(sentences) / seq_total_time

    print(f"Embedding time: {seq_embed_time:.2f}s")
    print(f"FAISS indexing time: {seq_faiss_time:.2f}s")
    print(f"Total time: {seq_total_time:.2f}s")
    print(f"Throughput: {seq_throughput:.1f} sentences/sec")

    # 10 Cores Ray distributed benchmark (CPU only)
    print("\n=== 10 Cores Ray Distributed (CPU Only) Ingestion ===")

    # Clear FAISS before Ray test
    print("Clearing FAISS data...")
    clear_faiss_data()
    vector_store_ray = VectorStore()

    # Time Ray initialization and worker setup
    print("Initializing Ray and workers...")
    init_start = time.time()
    pipeline_10_cores = DistributedIngestionPipeline(
        num_workers=10, batch_size=batch_size
    )
    init_time = time.time() - init_start
    print(f"Initialization time: {init_time:.2f}s\n")

    # Time only the embedding work
    core10_start = time.time()
    core10_embeddings, core10_embed_time = (
        pipeline_10_cores.embed_sentences_distributed(sentences)
    )
    core10_embed_total = time.time() - core10_start

    # Index to FAISS
    print("Indexing to FAISS...")
    faiss_ray_start = time.time()
    vector_store_ray.index_document(
        doc_id="benchmark_ray",
        filename="benchmark_distributed.txt",
        file_type="txt",
        sentences=sentences,
        paragraph_indices=paragraph_indices,
        embeddings=core10_embeddings,
    )
    core10_faiss_time = time.time() - faiss_ray_start
    core10_total_time = core10_embed_time + core10_faiss_time
    core10_throughput = len(sentences) / core10_total_time

    print(
        f"Embedding time (excluding init): {core10_embed_total:.2f}s (embedding: {core10_embed_time:.2f}s)"
    )
    print(f"FAISS indexing time: {core10_faiss_time:.2f}s")
    print(f"Total time (excluding init): {core10_total_time:.2f}s")
    print(f"Total time (including init): {init_time + core10_total_time:.2f}s")
    print(f"Throughput: {core10_throughput:.1f} sentences/sec")

    pipeline_10_cores.shutdown()

    # Clear FAISS after benchmark
    print("\nClearing FAISS data...")
    clear_faiss_data()

    # Calculate metrics (excluding init)
    speedup_excl_init = seq_total_time / core10_total_time
    improvement_pct_excl_init = (
        (seq_total_time - core10_total_time) / seq_total_time * 100
    )

    # Calculate metrics (including init)
    total_ray_time = init_time + core10_total_time
    speedup_incl_init = seq_total_time / total_ray_time
    improvement_pct_incl_init = (seq_total_time - total_ray_time) / seq_total_time * 100

    print("\n=== Results (Excluding Initialization) ===")
    print(f"Speedup: {speedup_excl_init:.2f}x")
    print(f"Latency Reduction: {improvement_pct_excl_init:.1f}%")
    print(f"Throughput Increase: {core10_throughput/seq_throughput:.2f}x")

    print("\n=== Results (Including Initialization) ===")
    print(f"Speedup: {speedup_incl_init:.2f}x")
    print(f"Latency Reduction: {improvement_pct_incl_init:.1f}%")
    print(f"Sequential: {seq_total_time:.2f}s vs Ray total: {total_ray_time:.2f}s")

    return {
        "text_length": len(text),
        "sentences": len(sentences),
        "paragraphs": max(paragraph_indices) + 1 if paragraph_indices else 0,
        "sequential": {
            "time": seq_total_time,
            "embedding_time": seq_embed_time,
            "faiss_time": seq_faiss_time,
            "throughput": seq_throughput,
            "device": embedding_service.device,
        },
        "ten_cores": {
            "time": core10_total_time,
            "embedding_time": core10_embed_time,
            "faiss_time": core10_faiss_time,
            "throughput": core10_throughput,
            "workers": 10,
            "batch_size": batch_size,
            "device": "cpu",
        },
        "improvement": {
            "speedup": speedup_excl_init,
            "latency_reduction_pct": improvement_pct_excl_init,
            "throughput_increase": core10_throughput / seq_throughput,
            "speedup_incl_init": speedup_incl_init,
            "latency_reduction_pct_incl_init": improvement_pct_incl_init,
        },
    }


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    batch_size = 128  # Default batch size
    multiplier = 100  # Default multiplier

    if len(sys.argv) > 1:
        batch_size = int(sys.argv[1])
    if len(sys.argv) > 2:
        multiplier = int(sys.argv[2])

    # Load confusable document
    confusable_path = os.path.join(
        os.path.dirname(__file__), "eval_philosophy_document.txt"
    )

    print(f"Loading eval_philosophy_document.txt...")
    with open(confusable_path, "r", encoding="utf-8") as f:
        base_text = f.read()

    # Multiply the text if requested
    sample_text = base_text * multiplier

    print(f"Loaded document with {len(base_text)} characters")
    if multiplier > 1:
        print(f"Multiplied by {multiplier}x = {len(sample_text)} total characters\n")
    else:
        print()

    print("Benchmarking: Single core CPU non-Ray vs 10 cores Ray (both CPU only)\n")
    results = benchmark_ingestion(sample_text, batch_size=batch_size)

    print(f"\n=== Final Summary ===")
    print(
        f"Sequential (non-Ray, CPU): {results['sequential']['time']:.2f}s, {results['sequential']['throughput']:.1f} sentences/sec"
    )
    print(
        f"10 Cores Ray (CPU): {results['ten_cores']['time']:.2f}s, {results['ten_cores']['throughput']:.1f} sentences/sec"
    )
    print(f"\nExcluding initialization:")
    print(f"  Speedup: {results['improvement']['speedup']:.2f}x")
    print(
        f"  Latency Reduction: {results['improvement']['latency_reduction_pct']:.1f}%"
    )
    print(f"\nIncluding initialization:")
    print(f"  Speedup: {results['improvement']['speedup_incl_init']:.2f}x")
    print(
        f"  Latency Reduction: {results['improvement']['latency_reduction_pct_incl_init']:.1f}%"
    )
