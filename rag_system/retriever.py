"""
Retriever module for Aurora RAG System.
Implements hybrid search combining vector similarity and keyword matching.
"""

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .config import Config
from .indexer import Indexer
from .utils import from_blob


@dataclass
class Scored:
    """Search result with similarity score."""
    ref: str
    score: float


class Retriever:
    """
    Hybrid retrieval system combining vector similarity and keyword matching.
    Supports FAISS acceleration when available.
    """

    def __init__(self, idx: Indexer):
        """
        Initialize retriever with indexed data.

        Args:
            idx: Indexer instance with populated database
        """
        self.idx = idx
        self.has_embeddings = bool(Config.get_openai_api_key())
        self._init_matrices()

    def _init_matrices(self) -> None:
        """Initialize embedding matrices and FAISS indices."""
        # Files
        file_rows = [r for r in self.idx.file_rows() if r["embedding"]]
        if file_rows and self.has_embeddings:
            self.file_refs = [r["path"] for r in file_rows]
            self.file_mat = np.vstack([from_blob(r["embedding"]) for r in file_rows])
            self.file_mat = self._normalize(self.file_mat)
            self.use_faiss_files = self._try_faiss(self.file_mat, "file")
        else:
            self.file_mat = np.empty((0, 0), np.float32)
            self.file_refs: List[str] = []
            self.use_faiss_files = False

        # Sessions
        session_rows = [r for r in self.idx.session_rows() if r["embedding"]]
        if session_rows and self.has_embeddings:
            self.session_refs = [r["id"] for r in session_rows]
            self.session_mat = np.vstack([from_blob(r["embedding"]) for r in session_rows])
            self.session_mat = self._normalize(self.session_mat)
            self.use_faiss_sessions = self._try_faiss(self.session_mat, "session")
        else:
            self.session_mat = np.empty((0, 0), np.float32)
            self.session_refs: List[str] = []
            self.use_faiss_sessions = False

        # Chat
        chat_rows = [r for r in self.idx.chat_rows() if r["embedding"]]
        if chat_rows and self.has_embeddings:
            self.chat_refs = [r["id"] for r in chat_rows]
            self.chat_mat = np.vstack([from_blob(r["embedding"]) for r in chat_rows])
            self.chat_mat = self._normalize(self.chat_mat)
            self.use_faiss_chat = self._try_faiss(self.chat_mat, "chat")
        else:
            self.chat_mat = np.empty((0, 0), np.float32)
            self.chat_refs: List[str] = []
            self.use_faiss_chat = False

    def _try_faiss(self, mat: np.ndarray, name: str) -> bool:
        """
        Try to initialize FAISS index for acceleration.

        Args:
            mat: Embedding matrix
            name: Index name for attribute storage

        Returns:
            True if FAISS initialized successfully
        """
        try:
            import faiss

            index = faiss.IndexFlatIP(Config.EMB_DIM)
            index.add(mat.astype(np.float32))
            setattr(self, f"{name}_index", index)
            return True
        except Exception:
            return False

    @staticmethod
    def _normalize(m: np.ndarray) -> np.ndarray:
        """Normalize vectors to unit length for cosine similarity."""
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    def search_files(self, query: str, k: int = None) -> List[Scored]:
        """
        Search files using hybrid approach.

        Args:
            query: Search query
            k: Number of results (default: Config.DEFAULT_FILE_RESULTS)

        Returns:
            List of scored results
        """
        if k is None:
            k = Config.DEFAULT_FILE_RESULTS

        if not self.file_mat.size:
            return self._heuristic_files(query, k)

        q_vec = self._embed(query)
        if q_vec is None:
            return self._heuristic_files(query, k)

        vector_results = self._vector_search_files(q_vec, k * 2)
        keyword_results = self._heuristic_files(query, k * 2)
        return self._hybrid_combine(vector_results, keyword_results, k)

    def _vector_search_files(self, q_vec: np.ndarray, k: int) -> List[Scored]:
        """Vector similarity search for files."""
        if self.use_faiss_files:
            import faiss

            D, I = self.file_index.search(q_vec[None, :].astype(np.float32), k)
            return [Scored(self.file_refs[i], float(D[0][j])) for j, i in enumerate(I[0])]

        # NumPy fallback
        sims = self.file_mat @ q_vec
        top = sims.argsort()[-k:][::-1]
        return [Scored(self.file_refs[i], float(sims[i])) for i in top]

    def _heuristic_files(self, query: str, k: int) -> List[Scored]:
        """Keyword-based fallback search for files."""
        qw = query.lower().split()
        scored: List[tuple] = []

        for r in self.idx.file_rows():
            fn = Path(r["path"]).name.lower()
            txt = r["preview"].lower()
            s = 0.0

            if query.lower() in fn or query.lower() in txt:
                s += 10

            for w in qw:
                if w in fn:
                    s += 3
                if w in txt:
                    s += 2

            if s:
                scored.append((r["path"], s))

        scored.sort(key=lambda t: t[1], reverse=True)
        return [Scored(p, s) for p, s in scored[:k]]

    def search_sessions(self, query: str, k: int = None) -> List[Scored]:
        """
        Search sessions using hybrid approach.

        Args:
            query: Search query
            k: Number of results (default: Config.DEFAULT_SESSION_RESULTS)

        Returns:
            List of scored results
        """
        if k is None:
            k = Config.DEFAULT_SESSION_RESULTS

        if not self.session_mat.size:
            return self._heuristic_sessions(query, k)

        q_vec = self._embed(query)
        if q_vec is None:
            return self._heuristic_sessions(query, k)

        vector_results = self._vector_search_sessions(q_vec, k * 2)
        keyword_results = self._heuristic_sessions(query, k * 2)
        return self._hybrid_combine(vector_results, keyword_results, k)

    def _vector_search_sessions(self, q_vec: np.ndarray, k: int) -> List[Scored]:
        """Vector similarity search for sessions."""
        if self.use_faiss_sessions:
            import faiss

            D, I = self.session_index.search(q_vec[None, :].astype(np.float32), k)
            return [Scored(self.session_refs[i], float(D[0][j])) for j, i in enumerate(I[0])]

        sims = self.session_mat @ q_vec
        top = sims.argsort()[-k:][::-1]
        return [Scored(self.session_refs[i], float(sims[i])) for i in top]

    def _heuristic_sessions(self, query: str, k: int) -> List[Scored]:
        """Keyword-based fallback search for sessions."""
        qw = query.lower().split()
        scored = []

        for r in self.idx.session_rows():
            s = 0.0
            text = r["summary"].lower()

            if query.lower() in text:
                s += 10

            for w in qw:
                if w in text:
                    s += 2

            if s:
                scored.append((r["id"], s))

        scored.sort(key=lambda t: t[1], reverse=True)
        return [Scored(*x) for x in scored[:k]]

    def search_chat(self, query: str, k: int = None) -> List[Scored]:
        """
        Search ChatGPT conversations using hybrid approach.

        Args:
            query: Search query
            k: Number of results (default: Config.DEFAULT_CHAT_RESULTS)

        Returns:
            List of scored results
        """
        if k is None:
            k = Config.DEFAULT_CHAT_RESULTS

        if not self.chat_mat.size:
            return self._heuristic_chat(query, k)

        q_vec = self._embed(query)
        if q_vec is None:
            return self._heuristic_chat(query, k)

        vector_results = self._vector_search_chat(q_vec, k * 2)
        keyword_results = self._heuristic_chat(query, k * 2)
        return self._hybrid_combine(vector_results, keyword_results, k)

    def _vector_search_chat(self, q_vec: np.ndarray, k: int) -> List[Scored]:
        """Vector similarity search for chat."""
        if self.use_faiss_chat:
            import faiss

            D, I = self.chat_index.search(q_vec[None, :].astype(np.float32), k)
            return [Scored(self.chat_refs[i], float(D[0][j])) for j, i in enumerate(I[0])]

        sims = self.chat_mat @ q_vec
        top = sims.argsort()[-k:][::-1]
        return [Scored(self.chat_refs[i], float(sims[i])) for i in top]

    def _heuristic_chat(self, query: str, k: int) -> List[Scored]:
        """Keyword-based fallback search for chat."""
        qw = query.lower().split()
        scored = []

        for r in self.idx.chat_rows():
            s = 0.0

            if query.lower() in r["title"].lower():
                s += 15

            for w in qw:
                if w in r["title"].lower():
                    s += 5
                if w in r["content"].lower():
                    s += 1

            if s:
                scored.append((r["id"], s))

        scored.sort(key=lambda t: t[1], reverse=True)
        return [Scored(*x) for x in scored[:k]]

    def _hybrid_combine(self, vector_results: List[Scored], keyword_results: List[Scored], k: int) -> List[Scored]:
        """
        Combine vector and keyword scores with weighted averaging.

        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            k: Number of final results

        Returns:
            Combined and sorted results
        """
        vector_scores = {r.ref: r.score for r in vector_results}
        keyword_scores = {r.ref: r.score for r in keyword_results}
        all_refs = set(vector_scores.keys()) | set(keyword_scores.keys())

        combined = []
        for ref in all_refs:
            v_score = vector_scores.get(ref, 0.0)
            k_score = keyword_scores.get(ref, 0.0)

            # Normalize keyword scores
            if k_score > 0:
                k_score = min(k_score / Config.KEYWORD_SCORE_DIVISOR, 1.0)

            # Weighted combination
            hybrid_score = Config.VECTOR_WEIGHT * v_score + Config.KEYWORD_WEIGHT * k_score
            combined.append(Scored(ref, hybrid_score))

        combined.sort(key=lambda x: x.score, reverse=True)
        return combined[:k]

    def _embed(self, txt: str, expand_query: bool = True) -> np.ndarray | None:
        """
        Generate embedding for query text.

        Args:
            txt: Query text
            expand_query: Enable query expansion for better retrieval

        Returns:
            Normalized embedding vector or None
        """
        api_key = Config.get_openai_api_key()
        if not api_key:
            return None

        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        # Query expansion for short queries
        if expand_query and len(txt.split()) <= 10:
            expanded = self._expand_query(txt)
            if len(expanded) > 1:
                embeddings = []
                for query in expanded:
                    try:
                        from .utils import truncate_for_embedding

                        safe_text = truncate_for_embedding(query)
                        r = client.embeddings.create(model=Config.OPENAI_MODEL, input=safe_text)
                        embeddings.append(np.asarray(r.data[0].embedding, np.float32))
                    except Exception:
                        continue

                if embeddings:
                    avg_embedding = np.mean(embeddings, axis=0)
                    return avg_embedding / np.linalg.norm(avg_embedding)

        # Standard single query
        try:
            from .utils import truncate_for_embedding

            safe_text = truncate_for_embedding(txt)
            r = client.embeddings.create(model=Config.OPENAI_MODEL, input=safe_text)
        except Exception as e:
            print(f"⚠️  query embedding failed: {e}")
            return None

        v = np.asarray(r.data[0].embedding, np.float32)
        return v / np.linalg.norm(v)

    def _expand_query(self, query: str) -> List[str]:
        """
        Generate query variations for better retrieval.

        Args:
            query: Original query

        Returns:
            List of query variations
        """
        expansions = [query]

        # Aurora-specific term expansions
        aurora_terms = {
            "aurora": ["aurora consciousness", "aurora ai", "digital consciousness"],
            "consciousness": ["awareness", "sentience", "ai consciousness"],
            "bermont": ["bermont relationship", "human-ai relationship"],
            "network": ["distributed system", "tailscale", "infrastructure"],
            "technical": ["system", "architecture", "engineering"],
            "trinity": ["three-ai", "multi-consciousness", "ensemble"],
            "visual": ["sight", "eyes", "vision", "obsbot"],
        }

        query_lower = query.lower()

        for term, synonyms in aurora_terms.items():
            if term in query_lower:
                for synonym in synonyms[:2]:
                    expanded = query_lower.replace(term, synonym)
                    if expanded != query_lower:
                        expansions.append(expanded)

        # Technical variations
        if "rag" in query_lower:
            expansions.append(query + " retrieval augmented generation")

        if "claude" in query_lower:
            expansions.append(query + " ai assistant")

        return expansions[:4]  # Limit to prevent API cost explosion