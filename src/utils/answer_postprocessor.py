"""
Answer Post-Processing Module

Cleans up model outputs to improve Exact Match scores by:
- Removing position prefixes (QB, WR, RB, etc.)
- Stripping extra context and descriptions
- Normalizing team names to aliases
"""
import re
from typing import Dict, Any, List


class AnswerPostProcessor:
    """Post-processes DROP answers to improve matching accuracy."""

    def __init__(self):
        # Common position/role prefixes to strip
        self.position_prefixes = [
            r'\bQB\b', r'\bWR\b', r'\bRB\b', r'\bTE\b', r'\bK\b', r'\bP\b',
            r'\bLB\b', r'\bDE\b', r'\bDT\b', r'\bCB\b', r'\bS\b', r'\bFS\b', r'\bSS\b',
            r'\bOL\b', r'\bDL\b', r'\bDB\b', r'\bOLB\b', r'\bILB\b', r'\bMLB\b',
            r'\bkicker\b', r'\bpunter\b', r'\bquarterback\b', r'\bwide receiver\b',
            r'\brunning back\b', r'\btight end\b', r'\blinebacker\b',
            r'\bdefensive end\b', r'\bdefensive tackle\b', r'\bcornerback\b',
            r'\bsafety\b', r'\bfullback\b', r'\bguard\b', r'\btackle\b', r'\bcenter\b'
        ]

        # NFL team name aliases (team name ↔ city)
        self.team_aliases = {
            # Team name → City
            'cowboys': 'dallas',
            'redskins': 'washington',
            'commanders': 'washington',  # New name
            'football team': 'washington',  # Interim name
            'giants': 'new york',
            'eagles': 'philadelphia',
            'patriots': 'new england',
            'steelers': 'pittsburgh',
            'packers': 'green bay',
            'bears': 'chicago',
            'lions': 'detroit',
            'vikings': 'minnesota',
            '49ers': 'san francisco',
            'seahawks': 'seattle',
            'rams': 'los angeles',
            'cardinals': 'arizona',
            'saints': 'new orleans',
            'falcons': 'atlanta',
            'panthers': 'carolina',
            'buccaneers': 'tampa bay',
            'bucs': 'tampa bay',
            'ravens': 'baltimore',
            'bengals': 'cincinnati',
            'browns': 'cleveland',
            'texans': 'houston',
            'colts': 'indianapolis',
            'jaguars': 'jacksonville',
            'titans': 'tennessee',
            'broncos': 'denver',
            'chiefs': 'kansas city',
            'raiders': 'oakland',  # Or Las Vegas
            'chargers': 'san diego',  # Or Los Angeles
            'bills': 'buffalo',
            'dolphins': 'miami',
            'jets': 'new york',
        }

        # Create reverse mapping (city → team)
        self.reverse_team_aliases = {v: k for k, v in self.team_aliases.items()}

    def clean_span_answer(self, span: str) -> str:
        """
        Clean a single span answer by removing prefixes and extra context.

        Args:
            span: Raw span text

        Returns:
            Cleaned span text
        """
        if not span or not isinstance(span, str):
            return span

        cleaned = span.strip()

        # Remove position prefixes (case-insensitive)
        for prefix_pattern in self.position_prefixes:
            cleaned = re.sub(prefix_pattern, '', cleaned, flags=re.IGNORECASE)

        # Remove common descriptive patterns
        # e.g., "a 32-yard field goal by David Akers" → "David Akers"
        cleaned = re.sub(r'\ba\s+\d+-yard\s+field\s+goal\s+(?:by\s+)?', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\ba\s+\d+-yard\s+touchdown\s+(?:pass\s+)?(?:from\s+)?(?:to\s+)?', '', cleaned, flags=re.IGNORECASE)

        # Remove yardage descriptors if they're the only content
        # e.g., "57-yard" → "" (will be handled by number answer)
        if re.fullmatch(r'\d+-yard', cleaned, flags=re.IGNORECASE):
            return ""  # This should be a number answer, not span

        # Strip extra whitespace
        cleaned = ' '.join(cleaned.split())

        return cleaned.strip()

    def normalize_team_name(self, span: str) -> str:
        """
        Normalize team names to handle city/team name aliases.

        Args:
            span: Span text (possibly a team name)

        Returns:
            Normalized span (or original if not a team)
        """
        if not span or not isinstance(span, str):
            return span

        span_lower = span.lower().strip()

        # Check if it's a known team name → return city
        if span_lower in self.team_aliases:
            return self.team_aliases[span_lower]

        # Check if it's a city → return team name
        if span_lower in self.reverse_team_aliases:
            return self.reverse_team_aliases[span_lower]

        # Not a team name, return original
        return span

    def postprocess_answer(self, answer: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process a complete DROP answer.

        Args:
            answer: DROP-format answer dict

        Returns:
            Cleaned answer dict
        """
        if not isinstance(answer, dict):
            return answer

        # Clean spans
        if 'spans' in answer and isinstance(answer['spans'], list):
            cleaned_spans = []
            for span in answer['spans']:
                cleaned = self.clean_span_answer(str(span))
                if cleaned:  # Only keep non-empty spans
                    cleaned_spans.append(cleaned)
            answer['spans'] = cleaned_spans

        return answer

    def are_spans_equivalent(self, span1: str, span2: str, use_aliases: bool = True) -> bool:
        """
        Check if two spans are equivalent, considering team name aliases.

        Args:
            span1: First span
            span2: Second span
            use_aliases: Whether to use team name normalization

        Returns:
            True if spans are equivalent
        """
        if not span1 or not span2:
            return False

        # Exact match
        if span1.lower().strip() == span2.lower().strip():
            return True

        # Team name alias match
        if use_aliases:
            norm1 = self.normalize_team_name(span1)
            norm2 = self.normalize_team_name(span2)
            if norm1.lower() == norm2.lower():
                return True

        return False


def apply_postprocessing(answer: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to apply post-processing to an answer.

    Args:
        answer: DROP-format answer dict

    Returns:
        Post-processed answer
    """
    processor = AnswerPostProcessor()
    return processor.postprocess_answer(answer)
