"""
RAG System v5.0 - Modular Architecture
Developed for any operating system.

A retrieval-augmented generation system with hybrid search capabilities.
"""

__version__ = "5.0.0"
__author__ = "RAG System Contributors"

from .config import Config
from .indexer import Indexer
from .retriever import Retriever
from .prompt_builder import PromptBuilder
from .llm_runner import LLMRunner

__all__ = [
    "Config",
    "Indexer",
    "Retriever",
    "PromptBuilder",
    "LLMRunner",
]