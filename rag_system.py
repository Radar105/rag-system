#!/usr/bin/env python3
"""
RAG System v5.0 - Main Entry Point
Modular retrieval-augmented generation system with hybrid search.

Usage:
    ./rag_system.py stats
    ./rag_system.py search "query terms"
    ./rag_system.py query "question for Claude"
    ./rag_system.py rebuild
"""

from cli import main

if __name__ == "__main__":
    main()