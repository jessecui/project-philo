import torch
from sentence_transformers import CrossEncoder
from typing import List, Tuple, Dict, Any
import time


class CrossEncoderReranker:
    """Service for reranking paragraphs using a cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the cross-encoder reranker.

        Args:
            model_name: Name of the cross-encoder model to use.
                       Default is 'cross-encoder/ms-marco-MiniLM-L-6-v2' which is
                       fast and memory-efficient for reranking.
        """
        # Detect best available device
        if torch.backends.mps.is_available():
            device = "mps"
            print(
                "🚀 MPS (Metal Performance Shaders) detected - using GPU acceleration for reranking"
            )
        elif torch.cuda.is_available():
            device = "cuda"
            print("🚀 CUDA detected - using GPU acceleration for reranking")
        else:
            device = "cpu"
            print("Using CPU for reranking")

        print(f"Loading cross-encoder model: {model_name} on {device}")
        # Use sigmoid activation to normalize scores to [0, 1]
        self.model = CrossEncoder(
            model_name, device=device, activation_fn=torch.nn.Sigmoid()
        )
        self.device = device
        self.model_name = model_name
        print(f"Cross-encoder model loaded successfully")

    def rerank_paragraphs(
        self,
        query: str,
        paragraphs: List[Tuple[str, int, str]],
        top_n: int = 10,
    ) -> Tuple[List[Tuple[str, int, str, float]], float]:
        """
        Rerank paragraphs using cross-encoder scoring.

        Args:
            query: Search query text
            paragraphs: List of tuples (doc_id, para_idx, paragraph_text)
            top_n: Number of top results to return after reranking

        Returns:
            Tuple of (reranked_paragraphs, reranking_time) where:
            - reranked_paragraphs: List of tuples (doc_id, para_idx, paragraph_text, score)
                                   sorted by relevance score (highest first)
            - reranking_time: Time taken for reranking in seconds
        """
        if not paragraphs:
            return ([], 0.0)

        start_time = time.time()

        # Create query-paragraph pairs for scoring
        pairs = [(query, para_text) for _, _, para_text in paragraphs]

        # Get relevance scores from cross-encoder
        # Scores are normalized to [0, 1] due to sigmoid activation
        scores = self.model.predict(pairs)

        # Combine paragraphs with their scores
        reranked = [
            (doc_id, para_idx, para_text, float(score))
            for (doc_id, para_idx, para_text), score in zip(paragraphs, scores)
        ]

        # Sort by score (highest first) and take top_n
        reranked.sort(key=lambda x: x[3], reverse=True)
        reranked = reranked[:top_n]

        reranking_time = time.time() - start_time

        print(
            f"Reranked {len(paragraphs)} paragraphs in {reranking_time:.3f}s, "
            f"returning top {len(reranked)}"
        )

        return reranked, reranking_time

    def score_single_pair(self, query: str, text: str) -> float:
        """
        Score a single query-text pair.

        Args:
            query: Search query text
            text: Document text to score

        Returns:
            Relevance score normalized to [0, 1]
        """
        score = self.model.predict([(query, text)])
        return float(score[0])
