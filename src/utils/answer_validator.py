"""
Answer Validation Module

Validates model answers for type consistency, grounding, and sanity checks
before fusion to reduce hallucinations and improve precision.
"""
import re
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class AnswerValidator:
    """
    Validates DROP-format answers for correctness and grounding.
    """

    def __init__(self, strict_mode: bool = False):
        """
        Args:
            strict_mode: If True, apply stricter validation (may reduce recall)
        """
        self.strict_mode = strict_mode

    def validate_answer(
        self,
        answer: Dict[str, Any],
        question: str,
        context: str,
        answer_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate an answer and return validation result.

        Args:
            answer: DROP-format answer dict with 'number', 'spans', 'date'
            question: The original question
            context: The context passage
            answer_type: Expected answer type ('number', 'spans', 'date')

        Returns:
            Dict with 'is_valid' (bool), 'reason' (str), 'confidence_penalty' (float)
        """
        # Extract answer components
        number = answer.get('number', '')
        spans = answer.get('spans', [])
        date = answer.get('date', {})

        # Determine primary answer type
        has_number = str(number).strip() != ''
        has_spans = isinstance(spans, list) and len(spans) > 0
        has_date = isinstance(date, dict) and any(str(v).strip() for v in date.values())

        # Validation checks
        validation_result = {
            'is_valid': True,
            'reason': '',
            'confidence_penalty': 0.0
        }

        # Check 1: Type consistency
        if answer_type == 'number' and not has_number:
            validation_result['is_valid'] = False
            validation_result['reason'] = 'Number expected but not provided'
            validation_result['confidence_penalty'] = 0.5
            return validation_result

        # Check 2: Number format validation
        if has_number:
            num_validation = self._validate_number(str(number), question, context)
            if not num_validation['is_valid']:
                return num_validation

        # Check 3: Span grounding validation (only if context available)
        if has_spans and context and context.strip():
            span_validation = self._validate_spans(spans, context, self.strict_mode)
            if not span_validation['is_valid']:
                return span_validation

        # Check 4: Date format validation
        if has_date:
            date_validation = self._validate_date(date)
            if not date_validation['is_valid']:
                return date_validation

        # Check 5: Answer completeness
        if not (has_number or has_spans or has_date):
            validation_result['is_valid'] = False
            validation_result['reason'] = 'Empty answer (no number, spans, or date)'
            validation_result['confidence_penalty'] = 1.0

        return validation_result

    def _validate_number(
        self,
        number: str,
        question: str,
        context: str
    ) -> Dict[str, Any]:
        """Validate number format and sanity."""

        # Check if numeric
        try:
            num_value = float(number.replace(',', ''))
        except (ValueError, AttributeError):
            return {
                'is_valid': False,
                'reason': f'Number "{number}" is not numeric',
                'confidence_penalty': 0.8
            }

        # Sanity check: unrealistic values
        if abs(num_value) > 1e6:  # Unlikely for DROP questions
            logger.warning(f'Suspiciously large number: {num_value}')
            return {
                'is_valid': True if not self.strict_mode else False,
                'reason': f'Large number {num_value} (may be outlier)',
                'confidence_penalty': 0.3
            }

        # Check for negative numbers where inappropriate
        if num_value < 0 and 'difference' not in question.lower():
            logger.warning(f'Negative number without difference context: {num_value}')
            return {
                'is_valid': True,  # Allow but penalize
                'reason': f'Negative number {num_value} without explicit difference',
                'confidence_penalty': 0.2
            }

        return {'is_valid': True, 'reason': '', 'confidence_penalty': 0.0}

    def _validate_spans(
        self,
        spans: list,
        context: str,
        strict: bool = False
    ) -> Dict[str, Any]:
        """Validate spans are grounded in context."""

        if not isinstance(spans, list):
            return {
                'is_valid': False,
                'reason': 'Spans must be a list',
                'confidence_penalty': 0.8
            }

        context_lower = context.lower()
        ungrounded_spans = []

        for span in spans:
            span_str = str(span).strip()
            if not span_str:
                continue

            span_lower = span_str.lower()

            # Check if span appears in context (case-insensitive)
            # Allow partial matching for multi-word spans
            if strict:
                # Strict: exact match required
                if span_lower not in context_lower:
                    ungrounded_spans.append(span_str)
            else:
                # Lenient: check if main words appear
                main_words = [w for w in span_lower.split() if len(w) > 3]
                if main_words:
                    # At least 50% of significant words should appear
                    found_count = sum(1 for w in main_words if w in context_lower)
                    if found_count / len(main_words) < 0.5:
                        ungrounded_spans.append(span_str)
                elif span_lower not in context_lower:
                    # Short spans must match exactly
                    ungrounded_spans.append(span_str)

        if ungrounded_spans:
            penalty = 0.5 if len(ungrounded_spans) == len(spans) else 0.3
            return {
                'is_valid': False if strict else True,
                'reason': f'Spans not found in context: {ungrounded_spans}',
                'confidence_penalty': penalty
            }

        return {'is_valid': True, 'reason': '', 'confidence_penalty': 0.0}

    def _validate_date(self, date: Dict[str, str]) -> Dict[str, Any]:
        """Validate date format."""

        if not isinstance(date, dict):
            return {
                'is_valid': False,
                'reason': 'Date must be a dict with day/month/year',
                'confidence_penalty': 0.8
            }

        day = str(date.get('day', '')).strip()
        month = str(date.get('month', '')).strip()
        year = str(date.get('year', '')).strip()

        # Validate day (1-31)
        if day:
            try:
                day_int = int(day)
                if not (1 <= day_int <= 31):
                    return {
                        'is_valid': False,
                        'reason': f'Invalid day: {day} (must be 1-31)',
                        'confidence_penalty': 0.7
                    }
            except ValueError:
                return {
                    'is_valid': False,
                    'reason': f'Day "{day}" is not numeric',
                    'confidence_penalty': 0.7
                }

        # Validate month (1-12)
        if month:
            try:
                month_int = int(month)
                if not (1 <= month_int <= 12):
                    return {
                        'is_valid': False,
                        'reason': f'Invalid month: {month} (must be 1-12)',
                        'confidence_penalty': 0.7
                    }
            except ValueError:
                return {
                    'is_valid': False,
                    'reason': f'Month "{month}" is not numeric',
                    'confidence_penalty': 0.7
                }

        # Validate year (reasonable range: 500-2100)
        if year:
            try:
                year_int = int(year)
                if not (500 <= year_int <= 2100):
                    return {
                        'is_valid': False,
                        'reason': f'Invalid year: {year} (out of range 500-2100)',
                        'confidence_penalty': 0.5
                    }
            except ValueError:
                return {
                    'is_valid': False,
                    'reason': f'Year "{year}" is not numeric',
                    'confidence_penalty': 0.7
                }

        return {'is_valid': True, 'reason': '', 'confidence_penalty': 0.0}


def apply_validation_to_result(
    result: Dict[str, Any],
    question: str,
    context: str,
    validator: AnswerValidator
) -> Dict[str, Any]:
    """
    Apply validation to a model result and adjust confidence.

    Args:
        result: Model result dict with 'answer', 'confidence', etc.
        question: Original question
        context: Context passage
        validator: AnswerValidator instance

    Returns:
        Modified result with adjusted confidence or marked as invalid
    """
    if 'answer' not in result:
        return result

    answer = result['answer']
    original_conf = result.get('confidence', 0.5)

    # Validate
    validation = validator.validate_answer(
        answer=answer,
        question=question,
        context=context
    )

    # Apply validation result
    if not validation['is_valid']:
        logger.warning(
            f"Answer validation failed: {validation['reason']}. "
            f"Original confidence: {original_conf:.2f}"
        )
        # Mark as invalid or heavily penalize
        result['confidence'] = max(0.1, original_conf * (1 - validation['confidence_penalty']))
        result['validation_failed'] = True
        result['validation_reason'] = validation['reason']
    elif validation['confidence_penalty'] > 0:
        # Apply confidence penalty for warnings
        result['confidence'] = original_conf * (1 - validation['confidence_penalty'])
        result['validation_warning'] = validation['reason']
        logger.debug(
            f"Answer validation warning: {validation['reason']}. "
            f"Confidence: {original_conf:.2f} → {result['confidence']:.2f}"
        )

    return result
