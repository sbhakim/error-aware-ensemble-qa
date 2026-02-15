# src/utils/ensemble_helpers.py

import re
import logging
import numpy as np
import string
from typing import Dict, Any, Optional, List, Union, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


# --- NEW: Model Name Canonicalization ---

def canonical_model_key(name: str) -> str:
    """Creates a consistent, simplified key for a model name."""
    if not name: return "unknown"
    n = name.lower()
    if "meta-llama" in n or "llama-3.2-3b" in n:
        return "llama-3.2-3b"
    if "mistral" in n and "7b" in n:
        return "mistral-7b"
    if "gemma" in n and ("1.1" in n or "7b" in n):
        return "gemma-1.1-7b"
    return n  # Fallback to the lowercased name


# --- Refactored & Improved Utility Functions ---

def normalize_drop_number(value_str: Optional[Any]) -> Optional[float]:
    """
    Normalizes numbers (string, int, float) to float for comparison.
    Handles None, commas, and common number words.
    """
    if value_str is None or (isinstance(value_str, str) and not value_str.strip()):
        return None
    try:
        if isinstance(value_str, (int, float)):
            return float(value_str)

        s = str(value_str).replace(",", "").strip().lower()
        if not s:
            return None

        # Expanded number words can be added here if needed
        words = {
            "zero": 0.0, "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0,
            "five": 5.0, "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0, "ten": 10.0
        }
        if s in words:
            return words[s]

        if re.fullmatch(r'-?\d+(\.\d+)?', s):
            return float(s)

        match_clean_num = re.match(r'(-?\d+(?:\.\d+)?)', s)
        if match_clean_num:
            return float(match_clean_num.group(1))

        return None
    except (ValueError, TypeError):
        return None


def are_drop_values_equivalent(
    obj1: Dict[str, Any],
    obj2: Dict[str, Any],
    value_type: str,
    *args,
    **kwargs
) -> bool:
    """
    Compares structured DROP answer values for equivalence, treating empty answers as non-equivalent.

    Backward-compatible shim:
      - Accepts optional 4th positional arg (e.g., 'mode' or a boolean flag) from older/newer call sites.
      - Optional kwargs:
          mode: "strict" | "relaxed"   (default: "strict")
          tolerance: float             (default: 1e-6 for numbers)
          treat_empty_as_agree: bool   (default: False)
    """
    try:
        # Optional controls (tolerant to unexpected extra arg)
        mode = kwargs.get("mode", "strict")
        tolerance = float(kwargs.get("tolerance", 1e-6))
        treat_empty_as_agree = bool(kwargs.get("treat_empty_as_agree", False))

        # If a 4th positional arg is passed as a legacy flag/mode, try to interpret it
        if args:
            legacy = args[0]
            # Common patterns we’ve seen: a string mode or a boolean “treat empty as equal”
            if isinstance(legacy, str):
                parsed = legacy.strip().lower()
                if parsed:
                    mode = parsed
            elif isinstance(legacy, (int, bool)):
                treat_empty_as_agree = bool(legacy)

        if value_type == "number":
            n1 = normalize_drop_number(obj1.get("number"))
            n2 = normalize_drop_number(obj2.get("number"))
            # In strict mode, any None breaks agreement; in relaxed mode, allow both None iff explicitly requested
            if n1 is None or n2 is None:
                return treat_empty_as_agree and (n1 is None and n2 is None)
            return abs(n1 - n2) <= tolerance

        elif value_type == "spans":
            spans1 = set(str(s).strip().lower() for s in obj1.get("spans", []) if str(s).strip())
            spans2 = set(str(s).strip().lower() for s in obj2.get("spans", []) if str(s).strip())
            if not spans1 and not spans2:
                return treat_empty_as_agree  # default False -> no agreement on empty-empty
            return spans1 == spans2

        elif value_type == "date":
            date1 = obj1.get("date", {}) or {}
            date2 = obj2.get("date", {}) or {}
            if isinstance(date1, dict) and isinstance(date2, dict):
                is_d1_empty = not any(str(date1.get(k, '')).strip() for k in ['day', 'month', 'year'])
                is_d2_empty = not any(str(date2.get(k, '')).strip() for k in ['day', 'month', 'year'])
                if is_d1_empty and is_d2_empty:
                    return treat_empty_as_agree  # default False
                return all(
                    str(date1.get(k, '')).strip() == str(date2.get(k, '')).strip()
                    for k in ['day', 'month', 'year']
                )
            return False

    except Exception as e:
        logger.warning(f"Error during DROP value comparison for type '{value_type}': {e}")
    return False


# --- New Ensemble-Specific Components ---

class QueryFeatureExtractor:
    """Detects query characteristics that correlate with model-specific failures."""

    def __init__(self):
        self.temporal_weak_signals = [
            'before', 'after', 'earlier', 'later', 'first half', 'second half', 'regulation', 'overtime'
        ]
        self.explicit_periods = [
            'q1', 'q2', 'q3', 'q4', '1st quarter', '2nd quarter', 'third quarter', 'fourth quarter'
        ]
        self.arithmetic_operators = ['percentage', 'percent', 'ratio', 'average', 'per']

    def extract_features(self, query: str) -> dict:
        """Analyzes a query string and returns a dictionary of boolean features."""
        features = {
            'has_temporal_ambiguity': False,
            'requires_decimal_arithmetic': False,
            'complex_nested_structure': False
        }
        if not query:
            return features
        query_lower = query.lower()
        if any(signal in query_lower for signal in self.temporal_weak_signals) and not any(
            period in query_lower for period in self.explicit_periods
        ):
            features['has_temporal_ambiguity'] = True
        if any(op in query_lower for op in self.arithmetic_operators):
            features['requires_decimal_arithmetic'] = True
        if query_lower.count('and') >= 2 or query_lower.count(',') >= 3:
            features['complex_nested_structure'] = True
        return features


class EnsembleFuser:
    """Handles multi-model result fusion with various strategies."""

    def __init__(self, fusion_strategy: str = 'error_aware', weights: Optional[Dict] = None):
        self.strategy = fusion_strategy
        self.feature_extractor = QueryFeatureExtractor()
        self.base_weights = weights or {'llama-3.2-3b': 1.0, 'mistral-7b': 1.0, 'gemma-1.1-7b': 1.0}
        self._TOKEN_RE = re.compile(r"[a-z0-9]+")
        self._MONTHS = {m: i for i, m in enumerate(
            ["january", "february", "march", "april", "may", "june", "july",
             "august", "september", "october", "november", "december"], 1)}

    def fuse(self, query: str, model_results: Dict[str, Dict]) -> Dict:
        """Main entry point that dispatches to the correct fusion strategy."""
        if self.strategy == 'error_aware':
            return self._error_aware_fusion(query, model_results)
        elif self.strategy == 'confidence':
            return self._confidence_weighted_pick(model_results)
        else:
            return self._majority_vote_fusion(model_results)

    def _get_answer_key(self, answer: Optional[Dict[str, Any]]) -> Optional[Tuple]:
        """Creates a robust, hashable key for a structured DROP answer for voting."""
        if not isinstance(answer, dict):
            return None

        # Numerical key (rounded to integer for robustness)
        num = normalize_drop_number(answer.get("number"))
        if num is not None:
            return ('number', int(round(num)))

        # Span key (token-based, sorted, and case-insensitive)
        span_toks = []
        for s in answer.get("spans", []):
            span_toks += self._TOKEN_RE.findall(str(s).lower())
        if span_toks:
            return ('spans', tuple(sorted(set(span_toks))))

        # Date key (stable order, normalized month)
        date = answer.get("date", {}) or {}
        y = str(date.get("year", "")).strip()
        m_str = str(date.get("month", "")).strip().lower()
        mo = str(self._MONTHS.get(m_str, m_str))
        d = str(date.get("day", "")).strip()
        if any([y, mo, d]):
            return ("date", (y, mo, d))

        return None

    def _is_populated(self, answer: Optional[Dict]) -> bool:
        """Checks if a DROP answer object contains any substantive data."""
        if not isinstance(answer, dict):
            return False
        return self._get_answer_key(answer) is not None

    def _find_majority(self, model_results: Dict[str, Dict]) -> Tuple[Optional[Dict], int, List[str]]:
        """Finds a majority answer (2 or more) among model results."""
        answer_keys = [self._get_answer_key(res.get('answer')) for res in model_results.values()]
        valid_keys = [key for key in answer_keys if key is not None]
        if not valid_keys:
            return None, 0, []

        count = Counter(valid_keys)
        if not count:
            return None, 0, []

        majority_key, majority_count = count.most_common(1)[0]
        if majority_count >= 2:
            majority_answer_obj, agreeing_models = None, []
            for model_name, res in model_results.items():
                if self._get_answer_key(res.get('answer')) == majority_key:
                    if majority_answer_obj is None:
                        majority_answer_obj = res.get('answer')
                    agreeing_models.append(model_name)
            return majority_answer_obj, majority_count, agreeing_models

        return None, 0, []

    def _error_aware_fusion(self, query: str, model_results: Dict[str, Dict]) -> Dict:
        """Combines results using: unanimous -> majority -> error-aware routing."""
        # FIX: Use canonical model keys
        canon_results = {canonical_model_key(k): v for k, v in model_results.items()}
        valid_results = {
            k: v for k, v in canon_results.items()
            if self._is_populated(v.get('answer')) and v.get('confidence', 0) > 0
        }

        if not valid_results:
            return {'answer': None, 'confidence': 0.0, 'fusion_type': 'no_valid_results', 'status': 'error'}

        answer_keys = {self._get_answer_key(res['answer']) for res in valid_results.values()}
        if len(valid_results) > 1 and len(answer_keys) == 1:
            confidences = [r.get('confidence', 0.0) for r in valid_results.values()]
            return {
                'answer': next(iter(valid_results.values()))['answer'],
                'confidence': min(1.0, float(np.mean(confidences)) * 1.2),
                'fusion_type': 'unanimous',
                'status': 'success'
            }

        majority_answer, _, agreeing_models = self._find_majority(valid_results)
        if majority_answer:
            agreeing_confidences = [valid_results[model].get('confidence', 0.0) for model in agreeing_models]
            return {
                'answer': majority_answer,
                'confidence': float(np.mean(agreeing_confidences)) if agreeing_confidences else 0.0,
                'fusion_type': 'majority',
                'status': 'success'
            }

        return self._resolve_disagreement(query, valid_results)

    # --- NEW: tiny numeric-question prior to steer disagreement resolution on DROP ---
    def _looks_like_numeric_question(self, q: str) -> bool:
        """
        Returns True if the query strongly suggests a numeric answer (counts, measurements, scores).
        Minimal heuristic to improve DROP performance without wider refactors.
        """
        if not q:
            return False
        ql = q.lower().strip()
        if ql.startswith("how many") or ql.startswith("how much"):
            return True
        numeric_hints = [
            "yards", "yard", "points", "touchdowns", "tds",
            "wins", "losses", "catches", "receptions", "field goals", "fg",
            "minutes", "seconds", "yards longer", "yards farther",
            "percentage", "percent", "%"
        ]
        return any(h in ql for h in numeric_hints)

    def _resolve_disagreement(self, query: str, model_results: Dict[str, Dict]) -> Dict:
        """When models disagree, route based on error-type detection."""
        query_features = self.feature_extractor.extract_features(query)
        weights = self.base_weights.copy()

        reason = "Default: Confidence-weighted selection"
        if query_features['has_temporal_ambiguity']:
            if 'llama-3.2-3b' in weights:
                weights['llama-3.2-3b'] *= 0.5
            reason = "Down-weighted Llama due to temporal ambiguity"
        elif query_features['requires_decimal_arithmetic']:
            if 'mistral-7b' in weights:
                weights['mistral-7b'] *= 0.5
            reason = "Down-weighted Mistral due to decimal arithmetic"
        elif query_features['complex_nested_structure']:
            if 'gemma-1.1-7b' in model_results and model_results['gemma-1.1-7b'].get('confidence', 0.0) < 0.7:
                weights['gemma-1.1-7b'] *= 0.5
                reason = "Down-weighted Gemma due to complex structure and low confidence"

        weighted_confidences = {
            model: model_results[model].get('confidence', 0.0) * weights.get(model, 1.0)
            for model in model_results.keys()
        }

        # NEW: if the query looks numeric, prefer numeric-like answers if any exist
        if self._looks_like_numeric_question(query):
            numeric_like_types = {"number", "count", "extreme_value_numeric"}

            def _is_numeric_ans(ans: Dict[str, Any]) -> bool:
                if not isinstance(ans, dict):
                    return False
                # Prefer true numeric payloads; type name is secondary
                has_num = normalize_drop_number(ans.get('number')) is not None
                if has_num:
                    return True
                return str(ans.get('type', '')).lower() in numeric_like_types

            numeric_candidates = {
                m: r for m, r in model_results.items()
                if _is_numeric_ans(r.get('answer', {}))
            }
            if numeric_candidates:
                weighted_numeric = {
                    m: weighted_confidences.get(m, 0.0) for m in numeric_candidates.keys()
                }
                if weighted_numeric:
                    best_model_name = max(weighted_numeric, key=weighted_numeric.get)
                    return {
                        'answer': model_results[best_model_name].get('answer'),
                        'confidence': model_results[best_model_name].get('confidence', 0.0),
                        'weighted_confidence': weighted_numeric[best_model_name],
                        'fusion_type': 'error_aware_routing',
                        'selected_model': best_model_name,
                        'routing_reason': reason + " + numeric-prior",
                        'status': 'success'
                    }

        if not weighted_confidences:
            return {
                'answer': None,
                'confidence': 0.0,
                'fusion_type': 'disagreement_no_results',
                'status': 'error'
            }

        best_model_name = max(weighted_confidences, key=weighted_confidences.get)
        return {
            'answer': model_results[best_model_name].get('answer'),
            'confidence': model_results[best_model_name].get('confidence', 0.0),
            'weighted_confidence': weighted_confidences[best_model_name],
            'fusion_type': 'error_aware_routing',
            'selected_model': best_model_name,
            'routing_reason': reason,
            'status': 'success'
        }

    def _confidence_weighted_pick(self, model_results: Dict[str, Dict]) -> Dict:
        """Simple fusion: picks the answer with the highest confidence."""
        canon_results = {canonical_model_key(k): v for k, v in model_results.items()}
        valid_results = {k: v for k, v in canon_results.items() if self._is_populated(v.get('answer'))}
        if not valid_results:
            return {'answer': None, 'confidence': 0.0, 'fusion_type': 'no_valid_results', 'status': 'error'}

        best_model_name = max(valid_results, key=lambda k: valid_results[k].get('confidence', 0.0))
        best_result = valid_results[best_model_name]

        return {
            'answer': best_result['answer'],
            'confidence': best_result.get('confidence', 0.0),
            'fusion_type': 'confidence_pick',
            'selected_model': best_model_name,
            'status': 'success'
        }

    def _majority_vote_fusion(self, model_results: Dict[str, Dict]) -> Dict:
        """Returns the majority answer, or falls back to highest confidence if no majority."""
        canon_results = {canonical_model_key(k): v for k, v in model_results.items()}
        valid_results = {k: v for k, v in canon_results.items() if self._is_populated(v.get('answer'))}

        majority_answer, _, agreeing_models = self._find_majority(valid_results)
        if majority_answer:
            agreeing_confidences = [valid_results[model].get('confidence', 0.0) for model in agreeing_models]
            return {
                'answer': majority_answer,
                'confidence': float(np.mean(agreeing_confidences)) if agreeing_confidences else 0.0,
                'fusion_type': 'majority',
                'status': 'success'
            }
        else:
            return self._confidence_weighted_pick(valid_results)
