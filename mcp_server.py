#!/usr/bin/env python3
"""
MCP Server for RAG System v5.0

Exposes RAG search and retrieval capabilities via Model Context Protocol.
Enables any MCP client (Claude Code, etc.) to search indexed knowledge bases.
"""

import asyncio
import json
from typing import Any, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

from rag_system import Indexer, Retriever, PromptBuilder, Config


# Initialize RAG components
indexer = Indexer(quiet=True)
retriever = Retriever(indexer)


async def main():
    """Main MCP server entry point."""
    server = Server("rag-system")

    @server.list_resources()
    async def list_resources() -> list[Resource]:
        """List available RAG resources."""
        return [
            Resource(
                uri="rag://stats",
                name="RAG Index Statistics",
                mimeType="application/json",
                description="Statistics about indexed documents (sessions, files, chats)",
            ),
            Resource(
                uri="rag://config",
                name="RAG Configuration",
                mimeType="application/json",
                description="Current RAG system configuration and paths",
            ),
        ]

    @server.read_resource()
    async def read_resource(uri: str) -> str:
        """Read RAG resource by URI."""
        if uri == "rag://stats":
            stats = {
                "sessions": len(indexer.session_rows()),
                "files": len(indexer.file_rows()),
                "chatgpt": len(indexer.chat_rows()),
                "total": len(indexer.session_rows()) + len(indexer.file_rows()) + len(indexer.chat_rows()),
                "database": str(Config.INDEX_DB),
            }
            return json.dumps(stats, indent=2)

        elif uri == "rag://config":
            config = {
                "index_db": str(Config.INDEX_DB),
                "default_paths": [str(p) for p in Config.DEFAULT_PATHS],
                "session_dirs": [str(p) for p in Config.SESSION_DIRS],
                "embedding_model": Config.OPENAI_MODEL,
                "vector_weight": Config.VECTOR_WEIGHT,
                "keyword_weight": Config.KEYWORD_WEIGHT,
            }
            return json.dumps(config, indent=2)

        raise ValueError(f"Unknown resource: {uri}")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available RAG tools."""
        return [
            Tool(
                name="rag_search",
                description="Search the RAG index using hybrid vector + keyword search. Returns ranked results without LLM interaction.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query text",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results per category (default: 5)",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="rag_build_context",
                description="Build RAG context for a query. Returns formatted context with relevant documents, ready for LLM consumption.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query to build context for",
                        },
                        "deep": {
                            "type": "boolean",
                            "description": "Use deep mode with extended context (default: false)",
                            "default": False,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="rag_rebuild",
                description="Force a full rebuild of the RAG index. Scans filesystem and generates embeddings for new/changed documents.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Any) -> list[TextContent]:
        """Execute RAG tool by name."""
        if name == "rag_search":
            query = arguments["query"]
            max_results = arguments.get("max_results", 5)

            # Ensure index is up to date
            indexer.ensure_up_to_date()

            # Search all categories
            file_results = retriever.search_files(query, max_results)
            session_results = retriever.search_sessions(query, max_results)
            chat_results = retriever.search_chat(query, max_results)

            # Format results
            output = []

            if file_results:
                output.append("FILES:")
                for result in file_results:
                    output.append(f"  {result.score:.2f}  {result.ref}")
                output.append("")

            if session_results:
                output.append("SESSIONS:")
                for result in session_results:
                    output.append(f"  {result.score:.2f}  {result.ref}")
                output.append("")

            if chat_results:
                output.append("CHATGPT:")
                for result in chat_results:
                    output.append(f"  {result.score:.2f}  {result.ref}")

            if not output:
                output.append("No results found.")

            return [TextContent(type="text", text="\n".join(output))]

        elif name == "rag_build_context":
            query = arguments["query"]
            deep = arguments.get("deep", False)

            # Ensure index is up to date
            indexer.ensure_up_to_date()

            # Build RAG context
            context = PromptBuilder(retriever, deep=deep).build(query)

            return [TextContent(type="text", text=context)]

        elif name == "rag_rebuild":
            # Force full rebuild
            indexer.ensure_up_to_date(force=True)

            stats = {
                "sessions": len(indexer.session_rows()),
                "files": len(indexer.file_rows()),
                "chatgpt": len(indexer.chat_rows()),
            }

            output = f"Index rebuilt successfully\n"
            output += f"Sessions: {stats['sessions']}\n"
            output += f"Files: {stats['files']}\n"
            output += f"ChatGPT: {stats['chatgpt']}"

            return [TextContent(type="text", text=output)]

        raise ValueError(f"Unknown tool: {name}")

    # Run server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
