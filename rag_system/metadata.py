"""
Metadata extraction module for RAG System.
Extracts dates, entities, keyphrases, and other metadata from text.
"""

import re
import json
from datetime import datetime
from typing import List, Dict, Set, Tuple
from collections import Counter


class MetadataExtractor:
    """
    Extracts structured metadata from text for enhanced retrieval.
    Implements date extraction, entity recognition, and keyphrase extraction.
    """

    # Date pattern regexes
    DATE_PATTERNS = [
        # ISO format: 2025-04-21, 2025-04-21T10:30:00
        (r'\b(\d{4})-(\d{2})-(\d{2})(?:T\d{2}:\d{2}:\d{2})?(?:\.\d{3})?(?:Z)?\b', 'iso'),
        # Month DD, YYYY: April 21, 2025 or Apr 21, 2025
        (r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2}),?\s+(\d{4})\b', 'written'),
        # MM/DD/YYYY: 04/21/2025
        (r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b', 'slash'),
    ]

    # Month name to number mapping
    MONTH_MAP = {
        'jan': 1, 'january': 1,
        'feb': 2, 'february': 2,
        'mar': 3, 'march': 3,
        'apr': 4, 'april': 4,
        'may': 5,
        'jun': 6, 'june': 6,
        'jul': 7, 'july': 7,
        'aug': 8, 'august': 8,
        'sep': 9, 'september': 9,
        'oct': 10, 'october': 10,
        'nov': 11, 'november': 11,
        'dec': 12, 'december': 12,
    }

    # Common entity patterns - Customize for your domain
    ENTITY_PATTERNS = {
        'person': [
            r'\b(Claude|GPT|OpenAI|Anthropic)\b',
            r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b',  # Two-word proper names
        ],
        'hardware': [
            r'\b(Raspberry Pi|RTX \d{4}(?:\s?Ti)?|GTX \d{4}|AMD MI300X|Tesla [A-Z]\d+)\b',
            r'\b(server|workstation|node|cluster)\b',
        ],
        'software': [
            r'\b(Claude|Docker|Python|Node\.js|RAG|Kubernetes|PostgreSQL)\b',
            r'\b(GPT-\d+|Sonnet|Opus)\b',
        ],
    }

    # Stopwords for keyphrase extraction
    STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'should', 'could', 'may', 'might', 'can', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
    }

    def extract_dates(self, text: str) -> List[str]:
        """
        Extract dates from text and normalize to ISO format (YYYY-MM-DD).

        Args:
            text: Input text to extract dates from

        Returns:
            List of dates in YYYY-MM-DD format, sorted chronologically
        """
        dates = set()

        for pattern, format_type in self.DATE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)

            for match in matches:
                try:
                    if format_type == 'iso':
                        # Already in ISO format, just extract YYYY-MM-DD
                        dates.add(f"{match.group(1)}-{match.group(2)}-{match.group(3)}")

                    elif format_type == 'written':
                        # Convert "April 21, 2025" to "2025-04-21"
                        month_name = match.group(1).lower()[:3]
                        month = self.MONTH_MAP.get(month_name)
                        day = int(match.group(2))
                        year = int(match.group(3))
                        if month and 1 <= day <= 31 and 2020 <= year <= 2030:
                            dates.add(f"{year:04d}-{month:02d}-{day:02d}")

                    elif format_type == 'slash':
                        # Convert "04/21/2025" to "2025-04-21"
                        month = int(match.group(1))
                        day = int(match.group(2))
                        year = int(match.group(3))
                        if 1 <= month <= 12 and 1 <= day <= 31 and 2020 <= year <= 2030:
                            dates.add(f"{year:04d}-{month:02d}-{day:02d}")

                except (ValueError, IndexError):
                    continue

        return sorted(list(dates))

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text (people, hardware, software).

        Args:
            text: Input text to extract entities from

        Returns:
            Dictionary mapping entity types to lists of extracted entities
        """
        entities = {
            'person': [],
            'hardware': [],
            'software': [],
        }

        for entity_type, patterns in self.ENTITY_PATTERNS.items():
            found = set()
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    entity = match.group(0)
                    found.add(entity)
            entities[entity_type] = sorted(list(found))

        return entities

    def extract_keyphrases(self, text: str, top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Extract important keyphrases using simple frequency analysis.

        Args:
            text: Input text to extract keyphrases from
            top_n: Number of top keyphrases to return

        Returns:
            List of (phrase, count) tuples sorted by frequency
        """
        # Simple tokenization - lowercase and split on non-alphanumeric
        words = re.findall(r'\b[a-z][a-z0-9_+\-\.]+\b', text.lower())

        # Filter stopwords and very short words
        words = [w for w in words if w not in self.STOPWORDS and len(w) > 2]

        # Count frequencies
        counter = Counter(words)

        # Get top N
        return counter.most_common(top_n)

    def extract_all(self, text: str) -> Dict:
        """
        Extract all metadata from text.

        Args:
            text: Input text to process

        Returns:
            Dictionary containing dates, entities, keyphrases, and summary stats
        """
        dates = self.extract_dates(text)
        entities = self.extract_entities(text)
        keyphrases = self.extract_keyphrases(text, top_n=10)

        return {
            'dates': dates,
            'entities': entities,
            'keyphrases': [kp[0] for kp in keyphrases],  # Just the phrases
            'keyphrase_counts': dict(keyphrases),  # Phrase -> count mapping
            'date_count': len(dates),
            'entity_count': sum(len(v) for v in entities.values()),
        }

    def to_json(self, metadata: Dict) -> str:
        """Convert metadata dict to JSON string for storage."""
        return json.dumps(metadata, ensure_ascii=False)

    def from_json(self, json_str: str) -> Dict:
        """Parse JSON string back to metadata dict."""
        return json.loads(json_str)


# Singleton instance for easy import
extractor = MetadataExtractor()