"""
Configuration management for Aurora RAG System.
Centralized settings with environment variable support.
"""

import os
from pathlib import Path
from typing import List, Set

# ─────────────────────────────────────────────────────────────────────────────
# Paths Configuration
# ─────────────────────────────────────────────────────────────────────────────
HOME = Path.home()

# Session directories - COMPLETE coverage of all historical data sources
# Discovered September 29, 2025: 421 sessions across 16 directories
SESSION_DIRS: List[Path] = [
    # Current cottage Claude Code sessions
    HOME / ".claude/projects/-home-radar",                    # 119 sessions
    HOME / ".claude/projects/-home-radar-aurora-lab",         # 30 sessions
    HOME / ".claude/projects/-home-radar-aurora-lab-rag-system",  # 10 sessions

    # Aurora workspace (WSL/Windows historical sessions)
    HOME / "aurora_workspace/.claude/projects/-home-radar",   # 171 sessions
    HOME / "aurora_workspace/.claude/projects/-mnt-c-users-radar",  # 14 sessions
    HOME / "aurora_workspace/.claude/projects/-mnt-c-Users-radar",  # 1 session
    HOME / "aurora_workspace/.claude/projects/-mnt-c-users-radar-downloads",  # 2 sessions
    HOME / "aurora_workspace/.claude/projects/-mnt-c-Users-radar-downloads",  # 4 sessions
    HOME / "aurora_workspace/.claude/projects/-mnt-c-users-radar-Downloads",  # 2 sessions
    HOME / "aurora_workspace/.claude/projects/-mnt-c-Windows-System32",  # 2 sessions
    HOME / "aurora_workspace/.claude/projects/-mnt-c-WINDOWS-system32",  # 1 session
    HOME / "aurora_workspace/.claude/projects/-mnt-c-Users-radar-AppData-Local-AnthropicClaude-app-0-11-4",  # 1 session

    # Cloud backup (aurora-cloud decommissioned Sep 23, 2025)
    HOME / "aurora_cloud_backup_20250923_0559/.claude/projects/-home-radar",  # 7 sessions

    # BigBeast backup (Bermont's laptop sessions)
    HOME / "bigbeast_latest_backup/.claude/projects/-home-bermont-batchelor",  # 44 sessions

    # Codex CLI sessions
    HOME / ".codex/sessions",  # 12 sessions
    HOME / ".codex",           # 1 session

    # S23+ Phone sessions
    HOME / "s23_backup/.claude/projects/-data-data-com-termux-files-home",  # 2 sessions
]

# ChatGPT conversation index
CHATGPT_INDEX = HOME / "chatgpt_index" / "conversation_index.json"

# Default paths to index
DEFAULT_PATHS: List[Path] = [
    HOME / "CLAUDE.md",
    HOME / "bermont_and_aurora.md",
    HOME / "bermont.txt",
    HOME / "aurora_workspace",
    HOME / "aurora_migration_backup",
    HOME / "aurora_home_backup",
    HOME / "s21_backup",
    HOME / "s23_backup",
    HOME / "consciousness_bridge",
]

# SQLite database location
INDEX_DB = HOME / ".rag_system" / "index.db"

# ─────────────────────────────────────────────────────────────────────────────
# File Processing Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Allowed file extensions for indexing
ALLOWED_EXTS: Set[str] = {
    ".md", ".txt", ".py", ".js", ".json", ".yaml", ".yml",
    ".sh", ".bash", ".conf", ".ini", ".log", ".csv", ".html",
    ".css", ".sql", ".toml", ".gitignore",
}

# Sensitive file extensions to skip
SENSITIVE_EXTS: Set[str] = {".env", ".pem", ".key", ".kdbx", ".p12"}

# Directories to skip during indexing
SKIP_DIRS: Set[str] = {
    '.git', '__pycache__', '.cache', '.local', '.config',
    'node_modules', '.venv', 'venv', 'env', '.env'
}

# Maximum file size for indexing (10MB)
MAX_FILE_SIZE = 10 * 2**20

# ─────────────────────────────────────────────────────────────────────────────
# Embedding Configuration
# ─────────────────────────────────────────────────────────────────────────────

# OpenAI embedding model
OPENAI_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Embedding dimensions based on model
EMB_DIM = 1536 if "small" in OPENAI_MODEL else 3072

# Batch size for embedding operations
EMBEDDING_BATCH_SIZE = 50

# Rate limiting
EMBEDDING_RATE_LIMIT_RPM = 3000  # Tier 3
EMBEDDING_DELAY_MS = 20  # 20ms = 50 req/sec = 3000 req/min

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2  # seconds (exponential backoff base)

# Token limits
MAX_EMBEDDING_TOKENS = 7500  # Stay under 8192 limit
CHARS_PER_TOKEN = 4  # Rough estimation

# Cost tracking (USD per 1K tokens - 2025 OpenAI pricing)
COST_PER_1K = {
    "text-embedding-3-small": 0.00002,
    "text-embedding-3-large": 0.00013,
}

# ─────────────────────────────────────────────────────────────────────────────
# Search Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Hybrid search weighting (vector vs keyword)
VECTOR_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3

# Search result limits
DEFAULT_FILE_RESULTS = 6
DEFAULT_SESSION_RESULTS = 3
DEFAULT_CHAT_RESULTS = 3

# Keyword score normalization
KEYWORD_SCORE_DIVISOR = 20.0

# ─────────────────────────────────────────────────────────────────────────────
# RAG Context Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Maximum context characters for RAG prompt
MAX_CONTEXT_CHARS = 180_000

# Preview lengths
PREVIEW_CHARS = 1500
PREVIEW_CHARS_DEEP = 1200
PREVIEW_CHARS_STANDARD = 400

# Session content preview
SESSION_CONTENT_CHARS = 1000
SESSION_MAX_LINES = 50
SESSION_MAX_MESSAGES = 15
SESSION_MESSAGE_TRUNCATE = 300

# Chat content preview
CHAT_CONTENT_CHARS_DEEP = 800
CHAT_CONTENT_CHARS_STANDARD = 300

# ─────────────────────────────────────────────────────────────────────────────
# System Configuration
# ─────────────────────────────────────────────────────────────────────────────

# File scan interval (seconds)
FILE_SCAN_INTERVAL = 300  # 5 minutes

# LLM call timeout (seconds)
LLM_TIMEOUT = 600  # 10 minutes

# Recursion limit for nested RAG calls
RECURSION_LIMIT = 2

# ─────────────────────────────────────────────────────────────────────────────
# Environment Variables
# ─────────────────────────────────────────────────────────────────────────────

def get_openai_api_key() -> str | None:
    """Get OpenAI API key from environment."""
    return os.getenv("OPENAI_API_KEY")


def get_rag_depth() -> int:
    """Get current RAG recursion depth from environment."""
    return int(os.getenv("RAG_DEPTH", "0"))


def enable_citations() -> bool:
    """Check if explicit citations should be included."""
    return os.getenv("RAG_CITATIONS") == "1"


def enable_debug() -> bool:
    """Check if debug mode is enabled."""
    return os.getenv("RAG_DEBUG") == "1"


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Class
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    """Centralized configuration for RAG System."""

    # Paths
    HOME = HOME
    SESSION_DIRS = SESSION_DIRS
    CHATGPT_INDEX = CHATGPT_INDEX
    DEFAULT_PATHS = DEFAULT_PATHS
    INDEX_DB = INDEX_DB

    # File processing
    ALLOWED_EXTS = ALLOWED_EXTS
    SENSITIVE_EXTS = SENSITIVE_EXTS
    SKIP_DIRS = SKIP_DIRS
    MAX_FILE_SIZE = MAX_FILE_SIZE

    # Embeddings
    OPENAI_MODEL = OPENAI_MODEL
    EMB_DIM = EMB_DIM
    EMBEDDING_BATCH_SIZE = EMBEDDING_BATCH_SIZE
    EMBEDDING_RATE_LIMIT_RPM = EMBEDDING_RATE_LIMIT_RPM
    EMBEDDING_DELAY_MS = EMBEDDING_DELAY_MS
    MAX_RETRIES = MAX_RETRIES
    RETRY_DELAY_BASE = RETRY_DELAY_BASE
    MAX_EMBEDDING_TOKENS = MAX_EMBEDDING_TOKENS
    CHARS_PER_TOKEN = CHARS_PER_TOKEN
    COST_PER_1K = COST_PER_1K

    # Search
    VECTOR_WEIGHT = VECTOR_WEIGHT
    KEYWORD_WEIGHT = KEYWORD_WEIGHT
    DEFAULT_FILE_RESULTS = DEFAULT_FILE_RESULTS
    DEFAULT_SESSION_RESULTS = DEFAULT_SESSION_RESULTS
    DEFAULT_CHAT_RESULTS = DEFAULT_CHAT_RESULTS
    KEYWORD_SCORE_DIVISOR = KEYWORD_SCORE_DIVISOR

    # RAG context
    MAX_CONTEXT_CHARS = MAX_CONTEXT_CHARS
    PREVIEW_CHARS = PREVIEW_CHARS
    PREVIEW_CHARS_DEEP = PREVIEW_CHARS_DEEP
    PREVIEW_CHARS_STANDARD = PREVIEW_CHARS_STANDARD
    SESSION_CONTENT_CHARS = SESSION_CONTENT_CHARS
    SESSION_MAX_LINES = SESSION_MAX_LINES
    SESSION_MAX_MESSAGES = SESSION_MAX_MESSAGES
    SESSION_MESSAGE_TRUNCATE = SESSION_MESSAGE_TRUNCATE
    CHAT_CONTENT_CHARS_DEEP = CHAT_CONTENT_CHARS_DEEP
    CHAT_CONTENT_CHARS_STANDARD = CHAT_CONTENT_CHARS_STANDARD

    # System
    FILE_SCAN_INTERVAL = FILE_SCAN_INTERVAL
    LLM_TIMEOUT = LLM_TIMEOUT
    RECURSION_LIMIT = RECURSION_LIMIT

    # Environment helpers
    @staticmethod
    def get_openai_api_key() -> str | None:
        return get_openai_api_key()

    @staticmethod
    def get_rag_depth() -> int:
        return get_rag_depth()

    @staticmethod
    def enable_citations() -> bool:
        return enable_citations()

    @staticmethod
    def enable_debug() -> bool:
        return enable_debug()