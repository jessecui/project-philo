"""
Google AI Gemini service for generating answers from retrieved context.

This service uses Google's Gemini 3 Flash model to generate grounded answers
based on document excerpts retrieved from the RAG pipeline.
"""

import os
import time
import asyncio
from typing import List, AsyncGenerator, Dict, Any
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig
from app.services.vector_store import SearchResult


class GeminiGenerator:
    """Generate answers using Google AI Gemini 3 Flash with streaming."""

    def __init__(
        self,
        model_name: str = "gemini-3-flash-preview",
        temperature: float = 1.0,
        max_output_tokens: int = 8192,
    ):
        """
        Initialize the Gemini generator.

        Args:
            model_name: Gemini model to use (default: gemini-3-flash-preview)
            temperature: Sampling temperature (default: 1.0)
            max_output_tokens: Maximum tokens in response (default: 2048)
        """
        # Load API key from environment
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable must be set. "
                "Get your API key from https://aistudio.google.com/apikey"
            )

        # Initialize Google AI client
        print(f"🔧 Initializing Google AI...")
        print(f"   Model: {model_name}")

        # Create client with API key
        self.client = genai.Client(api_key=api_key)

        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        print(f"✓ Google AI initialized successfully")

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
            context_parts.append(f"\n[{i}] Document: {result.filename}, Paragraph {result.paragraph_idx}")

            if result.reranking_score is not None:
                context_parts.append(
                    f"    Relevance Score: {result.reranking_score:.4f}"
                )

            context_parts.append(f"\n    {result.paragraph_text}")

        return "\n".join(context_parts)

    def _create_system_prompt(self) -> str:
        """
        Create the system prompt for answer generation.

        Returns:
            System instruction string
        """
        return """You are a thoughtful philosophy guide. Answer questions using the provided source excerpts.

Guidelines:

1. **Explain clearly**: Translate philosophical ideas into plain, accessible language. If a quote uses archaic or dense phrasing, unpack what it means in modern terms.

2. **Be concise**: Aim for 2-3 paragraphs. Get to the point, then support it.

3. **Cite naturally**: Reference works and authors (e.g., "Lao Tzu suggests in the Tao Te Ching..."). Quote directly only when the original wording is especially powerful.

4. **Answer the question**: Lead with a clear response, then explain the reasoning from the sources.

5. **Stay grounded**: Only draw from the provided excerpts. If they don't fully address the question, say so briefly.

Tone: Warm but not gushing. Clear but not dry. Like explaining an interesting idea to a curious friend."""

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

        # Create the system prompt and user prompt
        system_prompt = self._create_system_prompt()
        user_prompt = f"{context}\n\n=== Question ===\n{query}\n\n=== Answer ===\n"

        # Configure generation
        config = GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature if temperature is not None else self.temperature,
            max_output_tokens=(
                max_output_tokens
                if max_output_tokens is not None
                else self.max_output_tokens
            ),
            top_p=0.95,
            top_k=40,
            thinking_config=ThinkingConfig(
                thinking_level="low"  # Low thinking for fast RAG responses
            ),
        )

        # Generate streaming response
        # Run sync generator in thread pool to avoid blocking event loop
        try:
            loop = asyncio.get_event_loop()

            def sync_stream():
                """Synchronous generator that yields chunks."""
                for chunk in self.client.models.generate_content_stream(
                    model=self.model_name,
                    contents=user_prompt,
                    config=config,
                ):
                    if chunk.text:
                        yield chunk.text

            # Use a queue to pass chunks from thread to async generator
            import queue
            import threading

            chunk_queue = queue.Queue()
            error_holder = [None]

            def run_sync_stream():
                try:
                    for chunk in sync_stream():
                        chunk_queue.put(chunk)
                    chunk_queue.put(None)  # Signal completion
                except Exception as e:
                    error_holder[0] = e
                    chunk_queue.put(None)

            # Start sync stream in background thread
            thread = threading.Thread(target=run_sync_stream)
            thread.start()

            # Yield chunks as they arrive
            while True:
                # Use asyncio-friendly polling
                while chunk_queue.empty():
                    await asyncio.sleep(0.01)

                chunk = chunk_queue.get()
                if chunk is None:
                    break
                yield chunk

            thread.join()

            if error_holder[0]:
                raise error_holder[0]

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback

            traceback.print_exc()
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

        # Create the system prompt and user prompt
        system_prompt = self._create_system_prompt()
        user_prompt = f"{context}\n\n=== Question ===\n{query}\n\n=== Answer ===\n"

        # Configure generation
        config = GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature if temperature is not None else self.temperature,
            max_output_tokens=(
                max_output_tokens
                if max_output_tokens is not None
                else self.max_output_tokens
            ),
            top_p=0.95,
            top_k=40,
            thinking_config=ThinkingConfig(
                thinking_budget=128  # Minimum thinking for 2.5 Pro (cannot disable)
            ),
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
