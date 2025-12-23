import faiss
import numpy as np
import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict


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
