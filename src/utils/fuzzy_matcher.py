"""
Fuzzy String Matching for Answer Comparison

Handles minor spelling variations in span answers using edit distance.
Conservative thresholds to avoid false matches.
"""
from typing import Optional


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein (edit) distance between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Edit distance (number of insertions/deletions/substitutions)
    """
    if not s1:
        return len(s2) if s2 else 0
    if not s2:
        return len(s1)

    # Create distance matrix
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1]   # substitution
                )

    return dp[m][n]


def fuzzy_match_spans(span1: str, span2: str,
                     max_distance: Optional[int] = None) -> bool:
    """
    Check if two spans are fuzzy matches (allowing minor spelling differences).

    Args:
        span1: First span
        span2: Second span
        max_distance: Maximum edit distance to consider a match.
                     If None, uses adaptive threshold based on length.

    Returns:
        True if spans are fuzzy matches
    """
    if not span1 or not span2:
        return False

    # Normalize to lowercase for comparison
    s1 = span1.strip().lower()
    s2 = span2.strip().lower()

    # Exact match - always accept
    if s1 == s2:
        return True

    # Calculate edit distance
    distance = levenshtein_distance(s1, s2)

    # Determine threshold based on length (conservative)
    if max_distance is None:
        # Adaptive threshold based on average length
        avg_len = (len(s1) + len(s2)) / 2
        if avg_len <= 4:
            # Very short strings: allow distance 1
            max_distance = 1
        elif avg_len <= 8:
            # Short strings (e.g., "Jon Kitna"): allow distance 1-2
            max_distance = 1 if avg_len <= 6 else 2
        else:
            # Longer strings: allow distance 2
            # But require at least 80% similarity
            max_distance = 2
            similarity = 1 - (distance / max(len(s1), len(s2)))
            if similarity < 0.80:
                return False

    return distance <= max_distance


def normalize_span_for_comparison(span: str) -> str:
    """
    Normalize a span for comparison (lowercase, strip).

    Args:
        span: Raw span text

    Returns:
        Normalized span
    """
    return span.strip().lower() if span else ""


def are_spans_fuzzy_equal(spans1: list, spans2: list,
                          allow_fuzzy: bool = True) -> bool:
    """
    Check if two span lists are equal, optionally using fuzzy matching.

    Args:
        spans1: First list of spans
        spans2: Second list of spans
        allow_fuzzy: Whether to use fuzzy matching

    Returns:
        True if span lists are equivalent
    """
    if not spans1 and not spans2:
        return True
    if not spans1 or not spans2:
        return False

    # Try exact match first
    norm1 = {normalize_span_for_comparison(s) for s in spans1}
    norm2 = {normalize_span_for_comparison(s) for s in spans2}
    if norm1 == norm2:
        return True

    # Try fuzzy matching if enabled
    if allow_fuzzy and len(spans1) == len(spans2):
        # For same-length span lists, check if all spans fuzzy match
        # Sort by length to pair similar spans
        sorted1 = sorted(spans1, key=len)
        sorted2 = sorted(spans2, key=len)

        matched = all(
            fuzzy_match_spans(s1, s2)
            for s1, s2 in zip(sorted1, sorted2)
        )
        if matched:
            return True

    return False
