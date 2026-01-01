import torch
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from app.utils.document_processor import DocumentProcessor


class EmbeddingService:
    """Service for generating text embeddings using HuggingFace models."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding service with a pre-trained model.

        Args:
            model_name: Name of the sentence-transformers model to use.
                       Default is 'all-MiniLM-L6-v2' which is lightweight and fast.
                       Other options:
                       - 'all-mpnet-base-v2' (higher quality, slower)
                       - 'paraphrase-multilingual-MiniLM-L12-v2' (multilingual)
        """
        # Detect best available device
        if torch.backends.mps.is_available():
            device = "mps"
            print(
                "🚀 MPS (Metal Performance Shaders) detected - using GPU acceleration"
            )
        elif torch.cuda.is_available():
            device = "cuda"
            print("🚀 CUDA detected - using GPU acceleration")
        else:
            device = "cpu"
            print("Using CPU for inference")

        print(f"Loading embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        print(
            f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}"
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for multiple texts.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        # Generate embeddings in batch (more efficient)
        embeddings = self.model.encode(texts, convert_to_numpy=True)

        # Convert to list of lists for JSON serialization
        return embeddings.tolist()

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors produced by this model.

        Returns:
            Integer dimension of embedding vectors
        """
        return self.model.get_sentence_embedding_dimension()

    def embed_by_sentence(
        self, text: str, document_processor: "DocumentProcessor"
    ) -> Tuple[List[str], List[int], List[List[float]]]:
        """
        Split text into sentences by paragraph and generate embeddings for each sentence.

        Args:
            text: Input text to split and embed
            document_processor: DocumentProcessor instance for text splitting

        Returns:
            Tuple of (sentences, paragraph_indices, embeddings) where:
            - sentences: List of sentence strings
            - paragraph_indices: List of integers where paragraph_indices[i] is the
                                paragraph index for sentences[i]
            - embeddings: List of embedding vectors where embeddings[i] is the
                         embedding for sentences[i]
        """
        if not text or not text.strip():
            return ([], [], [])

        # Split text into sentences and track paragraphs
        sentences, paragraph_indices = (
            document_processor.split_into_sentences_and_paragraphs(text)
        )

        # Handle case where no sentences were extracted
        if not sentences:
            return ([], [], [])

        # Generate embeddings for all sentences in batch (efficient)
        embeddings = self.embed_texts(sentences)

        return (sentences, paragraph_indices, embeddings)
