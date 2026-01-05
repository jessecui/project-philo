"""
Vertex AI Gemini service for generating answers from retrieved context.

This service uses Google's Gemini 2.5 Pro model to generate grounded answers
based on document excerpts retrieved from the RAG pipeline.
"""

import os
import time
from typing import List, AsyncGenerator, Dict, Any
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions
from app.services.vector_store import SearchResult


class VertexAIGenerator:
    """Generate answers using Vertex AI Gemini 2.5 Pro with streaming."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-pro",
        temperature: float = 0.7,
        max_output_tokens: int = 2048,
    ):
        """
        Initialize the Vertex AI generator.

        Args:
            model_name: Gemini model to use (default: gemini-2.5-pro)
            temperature: Sampling temperature (default: 0.7)
            max_output_tokens: Maximum tokens in response (default: 2048)
        """
        # Load configuration from environment
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("VERTEX_AI_LOCATION", "us-central1")
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        if not project:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT environment variable must be set. "
                "See .env.example for configuration details."
            )

        if credentials_path and not os.path.exists(credentials_path):
            raise ValueError(
                f"Service account key file not found: {credentials_path}. "
                "Check GOOGLE_APPLICATION_CREDENTIALS path."
            )

        # Initialize Vertex AI
        print(f"🔧 Initializing Vertex AI...")
        print(f"   Project: {project}")
        print(f"   Location: {location}")
        print(f"   Model: {model_name}")

        # Create client with Vertex AI configuration
        self.client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=HttpOptions(api_version="v1"),
        )

        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        print(f"✓ Vertex AI initialized successfully")

    def _format_context(self, search_results: List[SearchResult]) -> str:
        """
        Format search results into a structured context string for the model.

        Args:
            search_results: List of search results from vector store

        Returns:
            Formatted context string with numbered excerpts
        """
        context_parts = []
        context_parts.append("=== Retrieved Document Excerpts ===\n")

        for i, result in enumerate(search_results, 1):
            context_parts.append(f"\n[{i}] Document: {result.filename}")
            context_parts.append(f"    Paragraph: {result.paragraph_idx}")

            if result.reranking_score is not None:
                context_parts.append(
                    f"    Relevance Score: {result.reranking_score:.4f}"
                )

            # Include context paragraphs before (if available)
            if result.context_paragraphs_before:
                context_parts.append("\n    Context (before):")
                for ctx in result.context_paragraphs_before:
                    context_parts.append(f"    {ctx}")

            # Main paragraph
            context_parts.append(f"\n    Main Text:")
            context_parts.append(f"    {result.paragraph_text}")

            # Include context paragraphs after (if available)
            if result.context_paragraphs_after:
                context_parts.append("\n    Context (after):")
                for ctx in result.context_paragraphs_after:
                    context_parts.append(f"    {ctx}")

        return "\n".join(context_parts)

    def _create_system_prompt(self) -> str:
        """
        Create the system prompt for answer generation.

        Returns:
            System instruction string
        """
        return """You are a knowledgeable philosophy assistant helping users understand philosophical texts.

Your task is to answer questions based ONLY on the provided document excerpts. Follow these guidelines:

1. **Ground your answer in the texts**: Only use information from the provided excerpts. Do not add external knowledge or speculation.

2. **Cite your sources**: When referencing information, cite the excerpt number in brackets, e.g., [1], [2], etc.

3. **Be comprehensive**: If multiple excerpts discuss the question, synthesize them into a coherent answer.

4. **Maintain philosophical tone**: Write clearly but preserve the philosophical depth of the source material.

5. **Acknowledge limitations**: If the excerpts don't fully answer the question, acknowledge what's missing.

6. **Direct quotes**: When using direct quotes, use quotation marks and cite the source excerpt.

7. **No hallucination**: Never invent quotes, ideas, or information not present in the excerpts.

Format your response in clear paragraphs with proper citations."""

    async def stream_answer(
        self,
        query: str,
        search_results: List[SearchResult],
        temperature: float = None,
        max_output_tokens: int = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming answer based on the query and retrieved context.

        Args:
            query: User's question
            search_results: List of relevant document excerpts
            temperature: Override default temperature
            max_output_tokens: Override default max tokens

        Yields:
            Text chunks from the model
        """
        # Format the context from search results
        context = self._format_context(search_results)

        # Create the full prompt
        user_prompt = f"{context}\n\n=== Question ===\n{query}\n\n=== Answer ===\n"

        # Configure generation
        config = GenerateContentConfig(
            temperature=temperature if temperature is not None else self.temperature,
            max_output_tokens=(
                max_output_tokens
                if max_output_tokens is not None
                else self.max_output_tokens
            ),
            top_p=0.95,
            top_k=40,
        )

        # Generate streaming response
        try:
            for chunk in self.client.models.generate_content_stream(
                model=self.model_name,
                contents=user_prompt,
                config=config,
            ):
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(f"❌ {error_msg}")
            raise Exception(error_msg)

    def generate_answer(
        self,
        query: str,
        search_results: List[SearchResult],
        temperature: float = None,
        max_output_tokens: int = None,
    ) -> Dict[str, Any]:
        """
        Generate a complete (non-streaming) answer.

        Args:
            query: User's question
            search_results: List of relevant document excerpts
            temperature: Override default temperature
            max_output_tokens: Override default max tokens

        Returns:
            Dictionary with answer text and metadata
        """
        # Format the context from search results
        context = self._format_context(search_results)

        # Create the full prompt
        user_prompt = f"{context}\n\n=== Question ===\n{query}\n\n=== Answer ===\n"

        # Configure generation
        config = GenerateContentConfig(
            temperature=temperature if temperature is not None else self.temperature,
            max_output_tokens=(
                max_output_tokens
                if max_output_tokens is not None
                else self.max_output_tokens
            ),
            top_p=0.95,
            top_k=40,
        )

        # Generate response
        start_time = time.time()
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=config,
            )
            generation_time = time.time() - start_time

            return {
                "answer": response.text,
                "generation_time": generation_time,
                "model": self.model_name,
                "usage": {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count,
                },
            }

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(f"❌ {error_msg}")
            raise Exception(error_msg)
