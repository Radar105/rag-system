"""
Chunking module for Aurora RAG System.
Implements semantic chunking with overlap and dialogue turn preservation.
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    start_pos: int
    end_pos: int
    chunk_id: int
    parent_id: str
    chunk_type: str  # 'session', 'file', 'chat'


class SemanticChunker:
    """
    Implements semantic chunking for RAG documents.
    Follows OpenAI best practices: 1000 tokens with 10-20% overlap.
    """

    # Chunking parameters (OpenAI recommendations)
    CHARS_PER_TOKEN = 4  # Rough approximation
    TARGET_TOKENS = 1000  # Sweet spot for embeddings
    TARGET_CHARS = TARGET_TOKENS * CHARS_PER_TOKEN  # ~4000 chars
    OVERLAP_RATIO = 0.15  # 15% overlap
    OVERLAP_CHARS = int(TARGET_CHARS * OVERLAP_RATIO)  # ~600 chars

    # Session-specific patterns
    SESSION_MESSAGE_PATTERN = re.compile(
        r'\{"type":\s*"(?:user|assistant)"[^}]*"text":\s*"([^"]*(?:\\.[^"]*)*)"',
        re.DOTALL
    )

    def chunk_text(
        self,
        text: str,
        parent_id: str,
        chunk_type: str = 'generic'
    ) -> List[Chunk]:
        """
        Chunk text with overlap for context preservation.

        Args:
            text: Text to chunk
            parent_id: ID of parent document
            chunk_type: Type of content being chunked

        Returns:
            List of Chunk objects
        """
        if chunk_type == 'session':
            return self._chunk_session(text, parent_id)
        else:
            return self._chunk_generic(text, parent_id, chunk_type)

    def _chunk_generic(
        self,
        text: str,
        parent_id: str,
        chunk_type: str
    ) -> List[Chunk]:
        """
        Generic chunking with overlap for files and non-session content.
        """
        if len(text) <= self.TARGET_CHARS:
            # Small enough, return as single chunk
            return [Chunk(
                text=text,
                start_pos=0,
                end_pos=len(text),
                chunk_id=0,
                parent_id=parent_id,
                chunk_type=chunk_type
            )]

        chunks = []
        chunk_id = 0
        start = 0

        while start < len(text):
            # Calculate chunk boundaries
            end = min(start + self.TARGET_CHARS, len(text))

            # Try to break at sentence boundary if not at end
            if end < len(text):
                # Look for sentence ending within last 500 chars
                search_start = max(start, end - 500)
                sentence_ends = [
                    m.end() for m in re.finditer(r'[.!?]\s+', text[search_start:end])
                ]
                if sentence_ends:
                    # Break at last sentence in range
                    end = search_start + sentence_ends[-1]

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    start_pos=start,
                    end_pos=end,
                    chunk_id=chunk_id,
                    parent_id=parent_id,
                    chunk_type=chunk_type
                ))
                chunk_id += 1

            # Move to next chunk with overlap
            start = end - self.OVERLAP_CHARS if end < len(text) else end

        return chunks

    def _chunk_session(
        self,
        text: str,
        parent_id: str
    ) -> List[Chunk]:
        """
        Session-specific chunking that preserves dialogue turns.
        Groups user question + assistant response together.
        """
        # Parse session JSONL format
        messages = self._extract_session_messages(text)

        if not messages:
            # Fallback to generic chunking if parsing fails
            return self._chunk_generic(text, parent_id, 'session')

        # Group into dialogue turns (user + assistant pairs)
        dialogue_turns = self._group_dialogue_turns(messages)

        # Create chunks from dialogue turns
        return self._chunk_dialogue_turns(dialogue_turns, parent_id)

    def _extract_session_messages(self, text: str) -> List[Dict]:
        """
        Extract messages from Claude session JSONL format.
        Returns list of {role, text} dicts.
        """
        messages = []

        for line in text.split('\n'):
            if not line.strip():
                continue

            # Look for user/assistant messages
            if '"type":"user"' in line or '"type":"assistant"' in line:
                role = 'user' if '"type":"user"' in line else 'assistant'

                # Extract text field (handle escaped quotes)
                match = re.search(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"', line)
                if match:
                    text_content = match.group(1)
                    # Unescape common patterns
                    text_content = text_content.replace(r'\"', '"')
                    text_content = text_content.replace(r'\n', '\n')
                    text_content = text_content.replace(r'\\', '\\')

                    if text_content.strip():
                        messages.append({
                            'role': role,
                            'text': text_content.strip()
                        })

        return messages

    def _group_dialogue_turns(self, messages: List[Dict]) -> List[List[Dict]]:
        """
        Group messages into dialogue turns (Q&A pairs).
        Each turn = user message(s) + following assistant response.
        """
        turns = []
        current_turn = []

        for msg in messages:
            if msg['role'] == 'user':
                # Start new turn if we have accumulated messages
                if current_turn:
                    turns.append(current_turn)
                    current_turn = []
                current_turn.append(msg)
            else:  # assistant
                current_turn.append(msg)
                # Complete the turn after assistant response
                if current_turn:
                    turns.append(current_turn)
                    current_turn = []

        # Add any remaining messages
        if current_turn:
            turns.append(current_turn)

        return turns

    def _chunk_dialogue_turns(
        self,
        turns: List[List[Dict]],
        parent_id: str
    ) -> List[Chunk]:
        """
        Create chunks from dialogue turns, combining turns if they fit.
        """
        chunks = []
        chunk_id = 0
        current_chunk_text = []
        current_chunk_size = 0
        start_pos = 0

        for turn in turns:
            # Calculate turn size
            turn_text = '\n\n'.join(f"{msg['role']}: {msg['text']}" for msg in turn)
            turn_size = len(turn_text)

            # Check if adding this turn exceeds target
            if current_chunk_size + turn_size > self.TARGET_CHARS and current_chunk_text:
                # Save current chunk
                chunk_text = '\n\n'.join(current_chunk_text)
                chunks.append(Chunk(
                    text=chunk_text,
                    start_pos=start_pos,
                    end_pos=start_pos + len(chunk_text),
                    chunk_id=chunk_id,
                    parent_id=parent_id,
                    chunk_type='session'
                ))
                chunk_id += 1

                # Start new chunk with overlap (include last turn for context)
                if len(current_chunk_text) > 1:
                    current_chunk_text = [current_chunk_text[-1], turn_text]
                    current_chunk_size = len(current_chunk_text[0]) + turn_size
                else:
                    current_chunk_text = [turn_text]
                    current_chunk_size = turn_size

                start_pos += len(chunk_text) - len(current_chunk_text[0])
            else:
                # Add turn to current chunk
                current_chunk_text.append(turn_text)
                current_chunk_size += turn_size

        # Add final chunk
        if current_chunk_text:
            chunk_text = '\n\n'.join(current_chunk_text)
            chunks.append(Chunk(
                text=chunk_text,
                start_pos=start_pos,
                end_pos=start_pos + len(chunk_text),
                chunk_id=chunk_id,
                parent_id=parent_id,
                chunk_type='session'
            ))

        return chunks


# Singleton instance
chunker = SemanticChunker()