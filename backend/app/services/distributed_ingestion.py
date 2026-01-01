import ray
import os
from typing import List, Tuple, Dict, Any
import time
from dataclasses import dataclass


@dataclass
class IngestionMetrics:
    """Metrics for distributed ingestion performance."""

    total_time: float
    parsing_time: float
    embedding_time: float
    indexing_time: float
    total_sentences: int
    total_paragraphs: int
    sentences_per_second: float
    improvement_vs_sequential: float = 0.0


@ray.remote
class EmbeddingWorker:
    """Ray worker for distributed embedding generation."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding worker with model.

        Args:
            model_name: Name of the sentence-transformers model
        """
        from sentence_transformers import SentenceTransformer

        worker_id = ray.get_runtime_context().get_worker_id()

        # Force CPU for Ray workers to avoid MPS contention
        # Multiple processes competing for MPS causes slowdown
        # CPU parallelism is more efficient for distributed workloads
        device = "cpu"
        print(f"[Worker {worker_id}] Loading model: {model_name} on CPU")
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        print(f"[Worker {worker_id}] Model loaded")

    def embed_batch(self, sentences: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of sentences.

        Args:
            sentences: List of sentences to embed

        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(sentences, convert_to_numpy=True)
        return embeddings.tolist()


class DistributedIngestionPipeline:
    """Distributed ingestion pipeline using Ray for parallel processing."""

    def __init__(
        self,
        num_workers: int = 8,
        batch_size: int = 32,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the distributed ingestion pipeline.

        Args:
            num_workers: Number of Ray workers for embedding (default: 8 for M4)
            batch_size: Batch size for each worker
            model_name: Sentence transformer model name
        """
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.model_name = model_name
        self.workers: List[ray.ObjectRef] = []

        # Disable tokenizer parallelism warning
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(num_cpus=num_workers, ignore_reinit_error=True)

        # Create worker pool
        self._initialize_workers()

    def _initialize_workers(self):
        """Initialize Ray worker pool."""
        print(f"Initializing {self.num_workers} CPU-based embedding workers...")
        self.workers = [
            EmbeddingWorker.remote(self.model_name) for _ in range(self.num_workers)
        ]
        # Warm up workers (load models)
        warmup_futures = [
            worker.embed_batch.remote(["test"]) for worker in self.workers
        ]
        ray.get(warmup_futures)
        print(f"✓ {self.num_workers} workers initialized and ready")

    def embed_sentences_distributed(
        self, sentences: List[str]
    ) -> Tuple[List[List[float]], float]:
        """
        Distribute sentence embedding across Ray workers.

        Args:
            sentences: List of sentences to embed

        Returns:
            Tuple of (embeddings, embedding_time)
        """
        if not sentences:
            return ([], 0.0)

        start_time = time.time()

        # Calculate optimal chunk size for even distribution
        chunk_size = max(1, len(sentences) // self.num_workers)

        # Split sentences into chunks for parallel processing
        sentence_chunks = [
            sentences[i : i + chunk_size] for i in range(0, len(sentences), chunk_size)
        ]

        print(
            f"Distributing {len(sentences)} sentences across {self.num_workers} workers "
            f"(~{chunk_size} sentences per worker)"
        )

        # Distribute work across workers in round-robin
        futures = []
        for i, chunk in enumerate(sentence_chunks):
            worker = self.workers[i % self.num_workers]
            future = worker.embed_batch.remote(chunk)
            futures.append(future)

        # Gather results
        print(f"Waiting for {len(futures)} workers to complete...")
        embedding_chunks = ray.get(futures)

        # Flatten results
        embeddings = []
        for chunk in embedding_chunks:
            embeddings.extend(chunk)

        embedding_time = time.time() - start_time
        throughput = len(sentences) / embedding_time
        print(
            f"Embedding completed: {len(sentences)} sentences in {embedding_time:.2f}s "
            f"({throughput:.1f} sentences/sec)"
        )

        return embeddings, embedding_time
        print(
            f"Embedding completed: {total_sentences} sentences in {embedding_time:.2f}s "
            f"({total_sentences/embedding_time:.1f} sentences/sec)"
        )

        return embeddings, embedding_time

    def process_document(
        self,
        sentences: List[str],
        paragraph_indices: List[int],
    ) -> Tuple[List[List[float]], IngestionMetrics]:
        """
        Process a document with distributed embedding generation.

        Args:
            sentences: List of sentence strings
            paragraph_indices: Paragraph index for each sentence

        Returns:
            Tuple of (embeddings, metrics)
        """
        total_start = time.time()

        # Generate embeddings (distributed)
        embeddings, embedding_time = self.embed_sentences_distributed(sentences)

        total_time = time.time() - total_start

        # Calculate metrics
        total_sentences = len(sentences)
        total_paragraphs = max(paragraph_indices) + 1 if paragraph_indices else 0
        sentences_per_second = (
            total_sentences / embedding_time if embedding_time > 0 else 0
        )

        metrics = IngestionMetrics(
            total_time=total_time,
            parsing_time=0.0,  # Parsing happens before this
            embedding_time=embedding_time,
            indexing_time=0.0,  # Indexing happens after this
            total_sentences=total_sentences,
            total_paragraphs=total_paragraphs,
            sentences_per_second=sentences_per_second,
        )

        return embeddings, metrics

    def benchmark_vs_sequential(
        self, sentences: List[str], sequential_embedding_service
    ) -> Dict[str, Any]:
        """
        Benchmark distributed vs sequential embedding performance.

        Args:
            sentences: Test sentences
            sequential_embedding_service: Sequential EmbeddingService instance

        Returns:
            Benchmark results
        """
        print("\n=== Benchmarking: Distributed vs Sequential ===")

        # Sequential benchmark
        print(f"\n[Sequential] Embedding {len(sentences)} sentences...")
        seq_start = time.time()
        seq_embeddings = sequential_embedding_service.embed_texts(sentences)
        seq_time = time.time() - seq_start
        seq_throughput = len(sentences) / seq_time

        print(
            f"[Sequential] Completed in {seq_time:.2f}s ({seq_throughput:.1f} sentences/sec)"
        )

        # Distributed benchmark
        print(f"\n[Distributed] Embedding {len(sentences)} sentences...")
        dist_embeddings, dist_time = self.embed_sentences_distributed(sentences)
        dist_throughput = len(sentences) / dist_time

        print(
            f"[Distributed] Completed in {dist_time:.2f}s ({dist_throughput:.1f} sentences/sec)"
        )

        # Calculate improvement
        speedup = seq_time / dist_time if dist_time > 0 else 0
        improvement_pct = (
            ((seq_time - dist_time) / seq_time * 100) if seq_time > 0 else 0
        )

        print(f"\n=== Results ===")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Latency Reduction: {improvement_pct:.1f}%")
        print(f"Throughput Increase: {dist_throughput/seq_throughput:.2f}x")

        return {
            "sentences_count": len(sentences),
            "sequential_time": seq_time,
            "distributed_time": dist_time,
            "sequential_throughput": seq_throughput,
            "distributed_throughput": dist_throughput,
            "speedup": speedup,
            "latency_reduction_pct": improvement_pct,
            "workers": self.num_workers,
        }

    def shutdown(self):
        """Shutdown Ray workers."""
        if ray.is_initialized():
            ray.shutdown()
            print("Ray workers shutdown")
