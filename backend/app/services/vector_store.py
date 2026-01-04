import faiss
import numpy as np
import json
import os
import time
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass, asdict, field

if TYPE_CHECKING:
    from app.services.reranker_service import CrossEncoderReranker


@dataclass
class DocumentMetadata:
    """Metadata for an indexed document."""

    doc_id: str
    filename: str
    file_type: str
    total_sentences: int
    total_paragraphs: int


@dataclass
class SentenceMetadata:
    """Metadata for an indexed sentence."""

    doc_id: str
    sentence_idx: int
    paragraph_idx: int
    sentence_text: str


@dataclass
class SearchResult:
    """Result from a similarity search."""

    doc_id: str
    filename: str
    paragraph_idx: int
    paragraph_text: str
    matched_sentences: List[str]
    similarity_scores: List[float]
    # Optional fields for 2-stage retrieval with context
    reranking_score: Optional[float] = None
    context_paragraphs_before: Optional[List[str]] = None
    context_paragraphs_after: Optional[List[str]] = None


class VectorStore:
    """FAISS-based vector store for sentence embeddings with paragraph hierarchy."""

    def __init__(self, data_dir: str = "data", embedding_dim: int = 384):
        """
        Initialize the vector store.

        Args:
            data_dir: Directory to store FAISS index and metadata
            embedding_dim: Dimension of embedding vectors (default 384 for all-MiniLM-L6-v2)
        """
        self.data_dir = data_dir
        self.embedding_dim = embedding_dim
        self.index_path = os.path.join(data_dir, "faiss.index")
        self.metadata_path = os.path.join(data_dir, "metadata.json")

        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        # Initialize or load FAISS index
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            self._load_metadata()
        else:
            # Create a new FAISS index (L2 distance)
            self.index = faiss.IndexFlatL2(embedding_dim)
            self.documents: Dict[str, DocumentMetadata] = {}
            self.sentences: List[SentenceMetadata] = []
            self.paragraph_cache: Dict[Tuple[str, int], str] = (
                {}
            )  # (doc_id, para_idx) -> full paragraph text

    def _load_metadata(self):
        """Load metadata from disk."""
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.documents = {
                k: DocumentMetadata(**v) for k, v in data["documents"].items()
            }
            self.sentences = [SentenceMetadata(**s) for s in data["sentences"]]
            self.paragraph_cache = {
                (parts[0], int(parts[1])): v
                for k, v in data["paragraph_cache"].items()
                if (parts := k.split("::"))
            }

    def _save_metadata(self):
        """Save metadata to disk."""
        data = {
            "documents": {k: asdict(v) for k, v in self.documents.items()},
            "sentences": [asdict(s) for s in self.sentences],
            "paragraph_cache": {
                f"{k[0]}::{k[1]}": v for k, v in self.paragraph_cache.items()
            },
        }
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _save_index(self):
        """Save FAISS index to disk."""
        faiss.write_index(self.index, self.index_path)

    def index_document(
        self,
        doc_id: str,
        filename: str,
        file_type: str,
        sentences: List[str],
        paragraph_indices: List[int],
        embeddings: List[List[float]],
    ) -> bool:
        """
        Index a document with its sentences and embeddings.

        Args:
            doc_id: Unique document identifier
            filename: Original filename
            file_type: File extension
            sentences: List of sentence strings
            paragraph_indices: Paragraph index for each sentence
            embeddings: Embedding vectors for each sentence

        Returns:
            True if successful
        """
        # Check if document already exists
        if doc_id in self.documents:
            return False

        # Store document metadata
        total_paragraphs = max(paragraph_indices) + 1 if paragraph_indices else 0
        self.documents[doc_id] = DocumentMetadata(
            doc_id=doc_id,
            filename=filename,
            file_type=file_type,
            total_sentences=len(sentences),
            total_paragraphs=total_paragraphs,
        )

        # Build paragraph cache (group sentences by paragraph)
        paragraph_texts: Dict[int, List[str]] = {}
        for sent, para_idx in zip(sentences, paragraph_indices):
            if para_idx not in paragraph_texts:
                paragraph_texts[para_idx] = []
            paragraph_texts[para_idx].append(sent)

        for para_idx, para_sentences in paragraph_texts.items():
            self.paragraph_cache[(doc_id, para_idx)] = " ".join(para_sentences)

        # Add sentence metadata
        for idx, (sentence, para_idx) in enumerate(zip(sentences, paragraph_indices)):
            self.sentences.append(
                SentenceMetadata(
                    doc_id=doc_id,
                    sentence_idx=idx,
                    paragraph_idx=para_idx,
                    sentence_text=sentence,
                )
            )

        # Add embeddings to FAISS index
        embeddings_array = np.array(embeddings, dtype=np.float32)
        self.index.add(embeddings_array)

        # Persist to disk
        self._save_metadata()
        self._save_index()

        return True

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        deduplicate_paragraphs: bool = True,
    ) -> List[SearchResult]:
        """
        Search for similar sentences and return results grouped by paragraph.

        Args:
            query_embedding: Embedding vector of the query
            top_k: Number of results to return
            deduplicate_paragraphs: If True, return only one result per paragraph

        Returns:
            List of search results with paragraph context
        """
        if self.index.ntotal == 0:
            return []

        # Convert query to numpy array
        query_array = np.array([query_embedding], dtype=np.float32)

        # Search FAISS index (get more results if deduplicating)
        search_k = top_k * 5 if deduplicate_paragraphs else top_k
        distances, indices = self.index.search(
            query_array, min(search_k, self.index.ntotal)
        )

        # Group results by (doc_id, paragraph_idx)
        paragraph_groups: Dict[Tuple[str, int], List[Tuple[str, float]]] = {}

        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for unfilled results
                continue

            sentence_meta = self.sentences[idx]
            key = (sentence_meta.doc_id, sentence_meta.paragraph_idx)

            # Convert L2 distance to similarity score (inverse)
            similarity = 1.0 / (1.0 + distance)

            if key not in paragraph_groups:
                paragraph_groups[key] = []

            paragraph_groups[key].append((sentence_meta.sentence_text, similarity))

        # Build search results
        results = []
        for (doc_id, para_idx), matches in paragraph_groups.items():
            doc_meta = self.documents.get(doc_id)
            if not doc_meta:
                continue

            paragraph_text = self.paragraph_cache.get((doc_id, para_idx), "")

            # Sort matches by similarity and get unique sentences
            matches.sort(key=lambda x: x[1], reverse=True)
            seen_sentences = set()
            unique_matches = []
            scores = []

            for sent, score in matches:
                if sent not in seen_sentences:
                    seen_sentences.add(sent)
                    unique_matches.append(sent)
                    scores.append(score)

            results.append(
                SearchResult(
                    doc_id=doc_id,
                    filename=doc_meta.filename,
                    paragraph_idx=para_idx,
                    paragraph_text=paragraph_text,
                    matched_sentences=unique_matches,
                    similarity_scores=scores,
                )
            )

        # Sort by best similarity score in each paragraph group
        results.sort(key=lambda x: max(x.similarity_scores), reverse=True)

        return results[:top_k]

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the index.
        Note: FAISS doesn't support deletion, so we rebuild the index.

        Args:
            doc_id: Document ID to delete

        Returns:
            True if successful
        """
        if doc_id not in self.documents:
            return False

        # Remove document metadata
        del self.documents[doc_id]

        # Find indices to keep
        indices_to_keep = [
            i for i, sent in enumerate(self.sentences) if sent.doc_id != doc_id
        ]

        # Rebuild index with remaining embeddings
        if indices_to_keep:
            # Extract embeddings to keep
            old_index = self.index
            embeddings_to_keep = []

            for idx in indices_to_keep:
                # Reconstruct vector from index
                vector = old_index.reconstruct(int(idx))
                embeddings_to_keep.append(vector)

            # Create new index
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            if embeddings_to_keep:
                embeddings_array = np.array(embeddings_to_keep, dtype=np.float32)
                self.index.add(embeddings_array)

            # Update sentences list
            self.sentences = [self.sentences[i] for i in indices_to_keep]
        else:
            # No documents left, create empty index
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.sentences = []

        # Clean up paragraph cache
        self.paragraph_cache = {
            k: v for k, v in self.paragraph_cache.items() if k[0] != doc_id
        }

        # Persist changes
        self._save_metadata()
        self._save_index()

        return True

    def list_documents(self) -> List[DocumentMetadata]:
        """
        List all indexed documents.

        Returns:
            List of document metadata
        """
        return list(self.documents.values())

    def get_document(self, doc_id: str) -> Optional[DocumentMetadata]:
        """
        Get metadata for a specific document.

        Args:
            doc_id: Document ID

        Returns:
            Document metadata or None if not found
        """
        return self.documents.get(doc_id)

    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_documents": len(self.documents),
            "total_sentences": self.index.ntotal,
            "total_paragraphs": len(self.paragraph_cache),
            "embedding_dimension": self.embedding_dim,
        }

    def get_paragraph_with_context(
        self, doc_id: str, para_idx: int, context_window: int = 2
    ) -> Tuple[Optional[str], List[str], List[str]]:
        """
        Get a paragraph with surrounding context paragraphs.

        Args:
            doc_id: Document ID
            para_idx: Paragraph index
            context_window: Number of paragraphs before and after to include (default: 2)

        Returns:
            Tuple of (paragraph_text, paragraphs_before, paragraphs_after)
            Returns (None, [], []) if paragraph not found
        """
        # Get the main paragraph
        main_paragraph = self.paragraph_cache.get((doc_id, para_idx))
        if not main_paragraph:
            return (None, [], [])

        # Get document metadata to know total paragraphs
        doc_meta = self.documents.get(doc_id)
        if not doc_meta:
            return (main_paragraph, [], [])

        # Get context before
        paragraphs_before = []
        for i in range(max(0, para_idx - context_window), para_idx):
            para = self.paragraph_cache.get((doc_id, i))
            if para:
                paragraphs_before.append(para)

        # Get context after
        paragraphs_after = []
        for i in range(
            para_idx + 1, min(doc_meta.total_paragraphs, para_idx + context_window + 1)
        ):
            para = self.paragraph_cache.get((doc_id, i))
            if para:
                paragraphs_after.append(para)

        return (main_paragraph, paragraphs_before, paragraphs_after)

    def search_with_reranking(
        self,
        query_text: str,
        query_embedding: List[float],
        reranker: "CrossEncoderReranker",
        top_k_faiss: int = 50,
        top_n_final: int = 10,
        context_window: int = 2,
    ) -> Tuple[List[SearchResult], Dict[str, float]]:
        """
        Two-stage retrieval: FAISS candidate retrieval + cross-encoder reranking.

        Args:
            query_text: Search query text (needed for cross-encoder)
            query_embedding: Query embedding vector (for FAISS stage)
            reranker: CrossEncoderReranker instance
            top_k_faiss: Number of candidates to retrieve from FAISS (default: 50)
            top_n_final: Number of final results after reranking (default: 10)
            context_window: Number of paragraphs before/after to include (default: 2)

        Returns:
            Tuple of (search_results, timing_info) where:
            - search_results: List of reranked SearchResult objects with context
            - timing_info: Dict with 'faiss_time', 'reranking_time', 'total_time'
        """
        total_start = time.time()

        # Stage 1: FAISS retrieval
        faiss_start = time.time()
        if self.index.ntotal == 0:
            return ([], {"faiss_time": 0.0, "reranking_time": 0.0, "total_time": 0.0})

        # Convert query to numpy array
        query_array = np.array([query_embedding], dtype=np.float32)

        # Retrieve top_k_faiss sentences from FAISS
        distances, indices = self.index.search(
            query_array, min(top_k_faiss, self.index.ntotal)
        )

        # Group by paragraph and build paragraph list
        paragraph_groups: Dict[Tuple[str, int], List[Tuple[str, float]]] = {}

        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            sentence_meta = self.sentences[idx]
            key = (sentence_meta.doc_id, sentence_meta.paragraph_idx)

            # Convert L2 distance to similarity score
            similarity = 1.0 / (1.0 + distance)

            if key not in paragraph_groups:
                paragraph_groups[key] = []

            paragraph_groups[key].append((sentence_meta.sentence_text, similarity))

        # Build paragraph list for reranking
        paragraphs_for_reranking = []
        for (doc_id, para_idx), matches in paragraph_groups.items():
            paragraph_text = self.paragraph_cache.get((doc_id, para_idx), "")
            if paragraph_text:
                paragraphs_for_reranking.append((doc_id, para_idx, paragraph_text))

        faiss_time = time.time() - faiss_start

        # Stage 2: Cross-encoder reranking
        if not paragraphs_for_reranking:
            return (
                [],
                {
                    "faiss_time": faiss_time,
                    "reranking_time": 0.0,
                    "total_time": time.time() - total_start,
                },
            )

        reranked_paragraphs, reranking_time = reranker.rerank_paragraphs(
            query_text, paragraphs_for_reranking, top_n=top_n_final
        )

        # Build final results with context expansion
        results = []
        for doc_id, para_idx, para_text, rerank_score in reranked_paragraphs:
            doc_meta = self.documents.get(doc_id)
            if not doc_meta:
                continue

            # Get context paragraphs
            main_para, paras_before, paras_after = self.get_paragraph_with_context(
                doc_id, para_idx, context_window
            )

            # Get matched sentences from FAISS stage
            matches = paragraph_groups.get((doc_id, para_idx), [])
            matches.sort(key=lambda x: x[1], reverse=True)
            matched_sentences = [sent for sent, _ in matches]
            similarity_scores = [score for _, score in matches]

            results.append(
                SearchResult(
                    doc_id=doc_id,
                    filename=doc_meta.filename,
                    paragraph_idx=para_idx,
                    paragraph_text=para_text,
                    matched_sentences=matched_sentences,
                    similarity_scores=similarity_scores,
                    reranking_score=rerank_score,
                    context_paragraphs_before=paras_before,
                    context_paragraphs_after=paras_after,
                )
            )

        total_time = time.time() - total_start

        timing_info = {
            "faiss_time": faiss_time,
            "reranking_time": reranking_time,
            "total_time": total_time,
        }

        return (results, timing_info)
