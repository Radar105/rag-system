"""
Utility functions for RAG System.
Helper functions for hashing, formatting, and data conversion.
"""

import numpy as np
from datetime import datetime
from hashlib import blake2s
from pathlib import Path
from typing import Sequence


def sha(fp: Path, n: int = 8192) -> str:
    """
    Compute Blake2s hash of file's first n bytes.

    Args:
        fp: Path to file
        n: Number of bytes to read (default 8192)

    Returns:
        Hex digest of hash
    """
    h = blake2s()
    with open(fp, "rb") as f:
        h.update(f.read(n))
    return h.hexdigest()


def now_iso() -> str:
    """
    Get current timestamp in ISO format.

    Returns:
        ISO formatted timestamp (seconds precision)
    """
    return datetime.now().isoformat(timespec="seconds")


def clamp(txt: str, n: int) -> str:
    """
    Truncate text to n characters with ellipsis indicator.

    Args:
        txt: Text to truncate
        n: Maximum length

    Returns:
        Original text or truncated version with indicator
    """
    if len(txt) <= n:
        return txt
    return txt[: n - 30] + "\n[…] (truncated)"


def to_blob(vec: Sequence[float]) -> bytes:
    """
    Convert vector to float32 binary blob for SQLite storage.

    Args:
        vec: Vector of floats

    Returns:
        Binary representation as float32 bytes
    """
    return np.asarray(vec, dtype=np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    """
    Convert binary blob back to numpy array.

    Args:
        blob: Binary data from SQLite

    Returns:
        Numpy array of float32 values
    """
    return np.frombuffer(blob, dtype=np.float32)


def estimate_tokens(text: str, chars_per_token: int = 4) -> int:
    """
    Estimate token count from character count.

    Args:
        text: Text to estimate
        chars_per_token: Rough chars per token ratio (default 4)

    Returns:
        Estimated token count
    """
    return len(text) // chars_per_token


def truncate_for_embedding(text: str, max_tokens: int = 7500, chars_per_token: int = 4) -> str:
    """
    Truncate text to stay under token limit for embeddings.

    Args:
        text: Text to truncate
        max_tokens: Maximum tokens allowed (default 7500)
        chars_per_token: Chars per token ratio (default 4)

    Returns:
        Truncated text that should fit within token limit
    """
    max_chars = max_tokens * chars_per_token
    return text[:max_chars] if len(text) > max_chars else text