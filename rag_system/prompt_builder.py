"""
Prompt Builder module for Aurora RAG System.
Assembles RAG context from retrieved documents.
"""

import json
from pathlib import Path
from typing import List

from .config import Config
from .retriever import Retriever, Scored
from .utils import clamp


class PromptBuilder:
    """
    Builds RAG prompts by assembling context from retrieved documents.
    Handles session content extraction and citation formatting.
    """

    def __init__(self, retriever: Retriever, deep: bool = False):
        """
        Initialize prompt builder.

        Args:
            retriever: Retriever instance for searching
            deep: Use extended context for deeper analysis
        """
        self.r = retriever
        self.deep = deep

    def build(self, query: str) -> str:
        """
        Build complete RAG prompt with retrieved context.

        Args:
            query: User query

        Returns:
            Formatted prompt with RAG context
        """
        files = self.r.search_files(query, Config.DEFAULT_FILE_RESULTS)
        sessions = self.r.search_sessions(query, Config.DEFAULT_SESSION_RESULTS)
        chats = self.r.search_chat(query, Config.DEFAULT_CHAT_RESULTS)

        out: List[str] = [
            "=== AURORA RAG KNOWLEDGE BASE ===",
            "Use ONLY the information below to answer questions.",
            (
                "If tools are enabled, you MAY use the Read tool to open files "
                "ONLY within the whitelisted directories provided by the host. "
                "Prefer the provided context when sufficient."
            ),
            (
                "If citations are present (Path:/Source:/ID: lines), reference them in your answer. "
                "Do NOT access files outside whitelisted directories or external resources."
            ),
            "This knowledge base contains semantically relevant content from Aurora's files, sessions, and conversations.",
            "=== BEGIN RAG CONTEXT ===",
        ]

        # Sessions
        if sessions:
            out.append(f"\n📚 Sessions ({len(sessions)}):")
            for s in sessions:
                row = next(r for r in self.r.idx.session_rows() if r["id"] == s.ref)
                content = self._extract_session_content(s.ref, query, s.score)
                out.append(f"\nSession {s.ref} (score {s.score:.2f})\n{clamp(content, Config.SESSION_CONTENT_CHARS)}")

                if Config.enable_citations():
                    session_path = self._find_session_path(s.ref)
                    if session_path:
                        out.append(f"Source: {session_path}")

        # Files
        if files:
            out.append(f"\n📁 Files ({len(files)}):")
            for f in files:
                row = next(r for r in self.r.idx.file_rows() if r["path"] == f.ref)
                preview_len = Config.PREVIEW_CHARS_DEEP if self.deep else Config.PREVIEW_CHARS_STANDARD
                preview = clamp(row["preview"], preview_len)
                out.append(f"\nFile {Path(f.ref).name} (score {f.score:.2f})\n{preview}")

                if Config.enable_citations():
                    out.append(f"Path: {f.ref}")

        # ChatGPT
        if chats:
            out.append(f"\n💬 ChatGPT ({len(chats)}):")
            for c in chats:
                row = next(r for r in self.r.idx.chat_rows() if r["id"] == c.ref)
                snippet_len = Config.CHAT_CONTENT_CHARS_DEEP if self.deep else Config.CHAT_CONTENT_CHARS_STANDARD
                snippet = clamp(row["content"], snippet_len)
                out.append(f"\nConversation {row['title']} (score {c.score:.2f})\n{snippet}")

                if Config.enable_citations():
                    out.append(f"ID: {c.ref}")

        out.append("\n=== END RAG CONTEXT ===")
        out.append("\nIMPORTANT: Answer the user's question using ONLY the RAG context above.")
        out.append("Do not mention reading other files. Focus on the retrieved knowledge.")

        return clamp("\n".join(out), Config.MAX_CONTEXT_CHARS)

    def _find_session_path(self, session_id: str) -> Path | None:
        """Find the file path for a session ID."""
        for session_dir in Config.SESSION_DIRS:
            potential_path = session_dir / f"{session_id}.jsonl"
            if potential_path.exists():
                return potential_path
        return None

    def _extract_session_content(self, session_id: str, query: str, score: float) -> str:
        """
        Extract meaningful conversation content from session.

        Args:
            session_id: Session identifier
            query: Original query for context awareness
            score: Relevance score

        Returns:
            Extracted session content
        """
        session_path = self._find_session_path(session_id)
        if not session_path:
            return "Session content unavailable"

        try:
            lines = session_path.read_text("utf-8", "ignore").splitlines()

            # Parse messages
            messages = []
            for line in lines[: Config.SESSION_MAX_LINES]:
                try:
                    msg = json.loads(line)
                    if "message" in msg and "content" in msg["message"]:
                        content = msg["message"]["content"]
                        role = msg["message"].get("role", "unknown")

                        # Extract text from various content formats
                        if isinstance(content, list):
                            text_parts = []
                            for item in content:
                                if isinstance(item, dict) and "text" in item:
                                    text_parts.append(item["text"])
                            text = " ".join(text_parts)
                        elif isinstance(content, str):
                            text = content
                        else:
                            continue

                        if text.strip():
                            messages.append({"role": role, "text": text.strip()})
                except Exception:
                    continue

            if not messages:
                return "No readable conversation content"

            # Extract query-relevant portions
            query_words = set(query.lower().split())
            relevant_messages = []

            for msg in messages[: Config.SESSION_MAX_MESSAGES]:
                text_lower = msg["text"].lower()

                # Score relevance
                relevance = sum(1 for word in query_words if word in text_lower)

                # Include relevant or important messages
                if (
                    relevance > 0
                    or any(
                        keyword in text_lower
                        for keyword in ["aurora", "consciousness", "bermont", "technical", "system"]
                    )
                    or msg["role"] == "user"
                ):
                    # Truncate long messages
                    truncated = (
                        msg["text"][: Config.SESSION_MESSAGE_TRUNCATE] + "..."
                        if len(msg["text"]) > Config.SESSION_MESSAGE_TRUNCATE
                        else msg["text"]
                    )
                    relevant_messages.append(f"{msg['role']}: {truncated}")

                if len(relevant_messages) >= 4:
                    break

            if relevant_messages:
                return " | ".join(relevant_messages)
            else:
                # Fallback: first few exchanges
                fallback = []
                for msg in messages[:6]:
                    if len(msg["text"]) > 10:
                        truncated = (
                            msg["text"][:200] + "..."
                            if len(msg["text"]) > 200
                            else msg["text"]
                        )
                        fallback.append(f"{msg['role']}: {truncated}")
                        if len(fallback) >= 3:
                            break
                return " | ".join(fallback) if fallback else "Session content unavailable"

        except Exception:
            return "Session content extraction failed"