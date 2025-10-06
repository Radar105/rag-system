#!/usr/bin/env python3
"""
Command-line interface for RAG System v5.0
"""

import argparse
import sys

from rag_system import Config, Indexer, Retriever, PromptBuilder, LLMRunner


def cmd_stats(args):
    """Show index statistics."""
    idx = Indexer(quiet=True)
    idx.ensure_up_to_date()

    print(f"📚 Sessions : {len(idx.session_rows())}")
    print(f"📁 Files    : {len(idx.file_rows())}")
    print(f"💬 ChatGPT  : {len(idx.chat_rows())}")


def cmd_rebuild(args):
    """Force full index rebuild."""
    idx = Indexer(quiet=False)
    idx.ensure_up_to_date(force=True)
    print("✅ Index rebuilt successfully")


def cmd_search(args):
    """Local search without Claude."""
    idx = Indexer(quiet=True)
    idx.ensure_up_to_date()

    retr = Retriever(idx)
    qtxt = " ".join(args.terms)

    print("\n📁 FILES")
    for sc in retr.search_files(qtxt, 10):
        print(f"{sc.score:5.2f}  {sc.ref}")

    print("\n📚 SESSIONS")
    for sc in retr.search_sessions(qtxt, 5):
        print(f"{sc.score:5.2f}  {sc.ref}")

    print("\n💬 CHATGPT")
    for sc in retr.search_chat(qtxt, 5):
        print(f"{sc.score:5.2f}  {sc.ref}")


def cmd_query(args):
    """Query with Claude using RAG context."""
    idx = Indexer(quiet=True)
    idx.ensure_up_to_date()

    question = " ".join(args.text)
    retriever = Retriever(idx)
    prompt = PromptBuilder(retriever, deep=args.deep).build(question)

    depth = Config.get_rag_depth()
    runner = LLMRunner(not args.no_tools)

    print("⏳ Contacting LLM...")
    response = runner.call(prompt + "\n\nUSER QUESTION: " + question, depth)
    print(response)


def main():
    """Main CLI entry point."""
    ap = argparse.ArgumentParser(
        prog="rag-system",
        description=(
            "RAG System v5.0 - Retrieval-Augmented Generation CLI\n"
            "Modular architecture with hybrid search capabilities."
        ),
        epilog=(
            "Environment Variables:\n"
            "  OPENAI_API_KEY        OpenAI key for embeddings (enables semantic search)\n"
            "  RAG_CITATIONS=1       Include explicit Path:/Source:/ID: lines\n"
            "  RAG_DEBUG=1           Enable LLM debug stream (verbose JSON)\n"
            "  RAG_DEPTH             Recursion guard (default 0; limit 2)\n\n"
            "Examples:\n"
            "  rag-system stats\n"
            "  rag-system search \"machine learning deployment\"\n"
            "  rag-system query \"What are ML best practices?\"\n"
            "  RAG_CITATIONS=1 rag-system query --deep \"explain RAG systems\"\n"
            "  rag-system query --no-tools \"answer without file access\""
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = ap.add_subparsers(dest="cmd", required=True)

    # Stats command
    sub.add_parser("stats", help="Show index statistics (counts of sessions/files/chat)")

    # Rebuild command
    sub.add_parser("rebuild", help="Force full re-scan and embed any missing content")

    # Search command
    s = sub.add_parser(
        "search",
        help="Local search only (no LLM)",
        description="Search the local RAG index and print ranked results without contacting an LLM.",
    )
    s.add_argument("terms", nargs="+", help="Search terms")

    # Query command
    q = sub.add_parser(
        "query",
        help="Ask with RAG context via LLM",
        description=(
            "Builds a RAG prompt from local files/sessions/chat and sends it to your configured LLM.\n"
            "Tools: Read only (whitelisted dirs) unless --no-tools is used.\n"
            "Use --deep for larger snippets; enable RAG_CITATIONS=1 to embed explicit sources."
        ),
    )
    q.add_argument("text", nargs="+", help="User question (plain text)")
    q.add_argument("--deep", action="store_true", help="Use longer previews/snippets in RAG context")
    q.add_argument("--no-tools", action="store_true", help="Disable tool use (prevents any file reads)")

    args = ap.parse_args()

    # Dispatch to command handlers
    if args.cmd == "stats":
        cmd_stats(args)
    elif args.cmd == "rebuild":
        cmd_rebuild(args)
    elif args.cmd == "search":
        cmd_search(args)
    elif args.cmd == "query":
        cmd_query(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)