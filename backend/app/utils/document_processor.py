from PyPDF2 import PdfReader
from docx import Document
import os
import nltk
from typing import List, Tuple


class DocumentProcessor:
    """Process different document types and extract text."""

    def __init__(self):
        """Initialize the document processor and download NLTK data."""
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

    def extract_text(self, file_path: str, file_ext: str) -> str:
        """
        Extract text from a document based on its file extension.

        Args:
            file_path: Path to the document file
            file_ext: File extension (e.g., '.pdf', '.txt', '.md', '.docx')

        Returns:
            Extracted text as a string
        """
        if file_ext == ".pdf":
            return self._extract_from_pdf(file_path)
        elif file_ext == ".txt" or file_ext == ".md":
            return self._extract_from_text(file_path)
        elif file_ext == ".docx":
            return self._extract_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file."""
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def _extract_from_text(self, file_path: str) -> str:
        """Extract text from a TXT or MD file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            try:
                with open(file_path, "r", encoding="latin-1") as f:
                    return f.read().strip()
            except Exception as e:
                raise Exception(f"Error reading text file: {str(e)}")
        except Exception as e:
            raise Exception(f"Error reading text file: {str(e)}")

    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from a DOCX file."""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from DOCX: {str(e)}")

    def split_into_sentences_and_paragraphs(
        self, text: str
    ) -> Tuple[List[str], List[int]]:
        """
        Split text into sentences and track which paragraph each sentence belongs to.

        Args:
            text: Input text to split

        Returns:
            Tuple of (sentences, paragraph_indices) where:
            - sentences: List of sentence strings
            - paragraph_indices: List of integers where paragraph_indices[i] is the
                                paragraph index (0-based) for sentences[i]
        """
        if not text or not text.strip():
            return ([], [])

        # Try splitting by double newlines first (standard paragraph delimiter)
        paragraphs = text.split("\n\n")

        # If no double newlines found, fall back to single newlines
        if len(paragraphs) == 1:
            paragraphs = text.split("\n")

        sentences = []
        paragraph_indices = []
        current_para_idx = 0

        for paragraph in paragraphs:
            # Skip empty or whitespace-only paragraphs
            if not paragraph.strip():
                continue

            # Tokenize paragraph into sentences
            para_sentences = nltk.sent_tokenize(paragraph.strip())

            # Filter out empty sentences and add to results
            for sentence in para_sentences:
                sentence = sentence.strip()
                if sentence:  # Only add non-empty sentences
                    sentences.append(sentence)
                    paragraph_indices.append(current_para_idx)

            # Only increment paragraph index if we added sentences
            if any(s.strip() for s in para_sentences):
                current_para_idx += 1

        return (sentences, paragraph_indices)
