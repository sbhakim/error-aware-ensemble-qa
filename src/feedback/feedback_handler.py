# src/feedback/feedback_handler.py

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict
import json
import numpy as np
from pathlib import Path


class FeedbackHandler:
    """
    Enhanced feedback handler that collects, analyzes, and processes feedback
    about the system's performance. This component is crucial for continuous
    improvement of the SymRAG system, particularly for multi-hop reasoning scenarios.
    """

    def __init__(self,
                 feedback_manager,
                 feedback_dir: str = "logs/feedback",
                 min_confidence: float = 0.3,
                 feedback_window: int = 100,
                 adaptation_threshold: float = 0.6,
                 interactive: bool = False):
        """
        Initialize the enhanced feedback handler with sophisticated tracking capabilities.

        Args:
            feedback_manager: Manager component for feedback storage
            feedback_dir: Directory for feedback-related files
            min_confidence: Minimum confidence threshold for feedback consideration
            feedback_window: Window size for feedback analysis
            adaptation_threshold: Threshold for system adaptation triggers
        """
        self.feedback_manager = feedback_manager
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)

        # Configuration parameters
        self.min_confidence = min_confidence
        self.feedback_window = feedback_window
        self.adaptation_threshold = adaptation_threshold
        self.interactive = interactive

        # Initialize logging
        self.logger = logging.getLogger("FeedbackHandler")
        self.logger.setLevel(logging.INFO)

        # Initialize feedback tracking structures
        self.feedback_stats = {
            'reasoning_paths': defaultdict(list),
            'performance_trends': defaultdict(list),
            'error_patterns': defaultdict(int),
            'adaptation_history': []
        }

        # Initialize feedback analysis components
        self._initialize_analysis_components()

        self.logger.info("FeedbackHandler initialized successfully")

    def _get_validated_rating(self, prompt: str) -> int:
        """
        Read and validate a feedback score in [1, 5].
        In non-interactive mode, returns a neutral score to keep pipelines non-blocking.
        """
        if not self.interactive:
            return 3

        raw = input(prompt).strip()
        try:
            rating = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Rating must be an integer in [1, 5], got: {raw}") from exc

        if not (1 <= rating <= 5):
            raise ValueError(f"Rating must be in [1, 5], got: {rating}")
        return rating

    def _initialize_analysis_components(self):
        """
        Initialize components for sophisticated feedback analysis.
        """
        # Define feedback categories and weights
        self.feedback_categories = {
            'accuracy': 0.4,
            'reasoning': 0.3,
            'completeness': 0.2,
            'efficiency': 0.1
        }

        # Initialize performance tracking
        self.performance_tracking = {
            'symbolic': {'scores': [], 'weights': []},
            'neural': {'scores': [], 'weights': []},
            'hybrid': {'scores': [], 'weights': []}
        }

        # Set up adaptation triggers
        self.adaptation_triggers = {
            'performance_threshold': 0.7,
            'error_rate_threshold': 0.2,
            'confidence_threshold': 0.6
        }

    def collect_feedback(self,
                         query: str,
                         result: Any,
                         reasoning_path: Optional[Dict] = None,
                         metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Collect and process comprehensive feedback about a query result.

        Args:
            query: The original query
            result: The system's response
            reasoning_path: Information about the reasoning process
            metadata: Additional context and processing information

        Returns:
            Dictionary containing processed feedback and recommendations
        """
        try:
            # Get user feedback with enhanced prompting
            feedback = self._collect_user_feedback(query, result, reasoning_path)

            # Process and analyze the feedback
            analyzed_feedback = self._analyze_feedback(
                feedback,
                reasoning_path,
                metadata
            )

            # Update feedback statistics
            self._update_feedback_stats(analyzed_feedback)

            # Generate improvement recommendations
            recommendations = self._generate_recommendations(analyzed_feedback)

            # Prepare comprehensive feedback report
            feedback_report = {
                'timestamp': datetime.now().isoformat(),
                'query_info': {
                    'query': query,
                    'reasoning_path': reasoning_path
                },
                'feedback_analysis': analyzed_feedback,
                'recommendations': recommendations,
                'adaptation_needed': self._check_adaptation_needed(analyzed_feedback)
            }

            # Store feedback for long-term analysis
            self._store_feedback(feedback_report)

            return feedback_report

        except Exception as e:
            self.logger.error(f"Error collecting feedback: {str(e)}")
            return {'error': str(e), 'status': 'failed'}

    def _collect_user_feedback(self,
                               query: str,
                               result: Any,
                               reasoning_path: Optional[Dict]) -> Dict[str, Any]:
        """
        Collect detailed user feedback with intelligent prompting.
        """
        feedback = {}

        # Determine appropriate feedback prompts based on query type
        prompts = self._generate_feedback_prompts(query, reasoning_path)

        for category, prompt in prompts.items():
            try:
                # Get numerical rating
                rating = self._get_validated_rating(prompt)

                # Get optional comment
                comment = input(f"Additional comments about {category} (optional): ").strip()

                feedback[category] = {
                    'rating': rating,
                    'comment': comment if comment else None
                }

            except ValueError as e:
                self.logger.warning(f"Invalid feedback input: {str(e)}")
                feedback[category] = {'rating': None, 'comment': str(e)}

        return feedback

    def _generate_feedback_prompts(self,
                                   query: str,
                                   reasoning_path: Optional[Dict]) -> Dict[str, str]:
        """
        Generate context-aware feedback prompts based on query characteristics.
        """
        prompts = {
            'accuracy': "Rate the accuracy of the answer (1-5): ",
            'completeness': "Rate the completeness of the response (1-5): "
        }

        # Add reasoning-specific prompts for multi-hop queries
        if reasoning_path and reasoning_path.get('hop_count', 0) > 1:
            prompts['reasoning'] = "Rate the clarity of the reasoning steps (1-5): "
            prompts['connection'] = "Rate how well the answer connects different facts (1-5): "

        return prompts

    def _analyze_feedback(self,
                          feedback: Dict[str, Any],
                          reasoning_path: Optional[Dict],
                          metadata: Optional[Dict]) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of collected feedback.
        """
        analysis = {
            'overall_score': self._calculate_overall_score(feedback),
            'category_scores': self._calculate_category_scores(feedback),
            'improvement_areas': self._identify_improvement_areas(feedback),
            'confidence': self._calculate_feedback_confidence(feedback)
        }

        # Add reasoning path analysis if available
        if reasoning_path:
            analysis['reasoning_analysis'] = self._analyze_reasoning_path(
                feedback,
                reasoning_path
            )

        # Include performance analysis
        analysis['performance_metrics'] = self._analyze_performance_metrics(
            feedback,
            metadata
        )

        return analysis

    def _calculate_overall_score(self, feedback: Dict[str, Any]) -> float:
        """
        Calculate weighted overall score from feedback categories.
        """
        weighted_scores = []
        total_weight = 0

        for category, weight in self.feedback_categories.items():
            if category in feedback:
                rating = feedback[category].get('rating')
                if rating is not None:
                    weighted_scores.append(rating * weight)
                    total_weight += weight

        if not weighted_scores:
            return 0.0

        return sum(weighted_scores) / total_weight if total_weight > 0 else 0.0

    def _identify_improvement_areas(self, feedback: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify specific areas needing improvement based on feedback.
        """
        improvements = []

        for category, data in feedback.items():
            rating = data.get('rating')
            if rating is not None and rating < 4:
                improvements.append({
                    'category': category,
                    'current_rating': rating,
                    'priority': self.feedback_categories.get(category, 0.1),
                    'comment': data.get('comment')
                })

        # Sort by priority and rating
        improvements.sort(key=lambda x: (x['priority'], -x['current_rating']))

        return improvements

    def _analyze_reasoning_path(self,
                                feedback: Dict[str, Any],
                                reasoning_path: Dict) -> Dict[str, Any]:
        """
        Analyze feedback specific to the reasoning path.
        """
        return {
            'path_type': reasoning_path.get('type', 'unknown'),
            'hop_effectiveness': self._calculate_hop_effectiveness(
                feedback,
                reasoning_path
            ),
            'path_confidence': reasoning_path.get('confidence', 0.0),
            'improvement_suggestions': self._generate_path_improvements(
                feedback,
                reasoning_path
            )
        }

    def _calculate_hop_effectiveness(self,
                                     feedback: Dict[str, Any],
                                     reasoning_path: Dict) -> float:
        """
        Calculate effectiveness score for multi-hop reasoning.
        """
        if 'reasoning' not in feedback or 'connection' not in feedback:
            return 0.0

        reasoning_score = feedback['reasoning'].get('rating', 0)
        connection_score = feedback['connection'].get('rating', 0)

        if not reasoning_score or not connection_score:
            return 0.0

        hop_count = reasoning_path.get('hop_count', 1)

        # Adjust score based on hop count (more hops = harder)
        base_score = (reasoning_score + connection_score) / 2
        return base_score * (1 + 0.1 * (hop_count - 1))

    def _generate_path_improvements(self,
                                    feedback: Dict[str, Any],
                                    reasoning_path: Dict) -> List[str]:
        """
        Generate specific suggestions for improving reasoning paths.
        """
        suggestions = []

        # Analyze hop structure
        if reasoning_path.get('hop_count', 0) > 3:
            suggestions.append("Consider simplifying reasoning path - too many hops")

        # Check confidence scores
        if reasoning_path.get('confidence', 1.0) < 0.7:
            suggestions.append("Improve confidence in reasoning steps")

        # Add feedback-based suggestions
        if 'reasoning' in feedback and feedback['reasoning'].get('rating', 5) < 4:
            suggestions.append("Clarify reasoning steps and connections")

        return suggestions

    def _update_feedback_stats(self, analyzed_feedback: Dict[str, Any]):
        """
        Update running statistics with new feedback data.
        """
        try:
            # Update reasoning path stats
            path_type = analyzed_feedback.get('reasoning_analysis', {}).get('path_type')
            if path_type:
                self.feedback_stats['reasoning_paths'][path_type].append(
                    analyzed_feedback['overall_score']
                )

            # Update performance trends
            self.feedback_stats['performance_trends']['overall'].append(
                analyzed_feedback['overall_score']
            )

            # Maintain window size
            self._maintain_stats_window()

        except Exception as e:
            self.logger.error(f"Error updating feedback stats: {str(e)}")

    def _maintain_stats_window(self):
        """
        Maintain the specified window size for feedback statistics.
        """
        for path_type in self.feedback_stats['reasoning_paths']:
            if len(self.feedback_stats['reasoning_paths'][path_type]) > self.feedback_window:
                self.feedback_stats['reasoning_paths'][path_type] = \
                    self.feedback_stats['reasoning_paths'][path_type][-self.feedback_window:]

        for metric in self.feedback_stats['performance_trends']:
            if len(self.feedback_stats['performance_trends'][metric]) > self.feedback_window:
                self.feedback_stats['performance_trends'][metric] = \
                    self.feedback_stats['performance_trends'][metric][-self.feedback_window:]

    def _check_adaptation_needed(self, analyzed_feedback: Dict[str, Any]) -> bool:
        """
        Determine if system adaptation is needed based on feedback analysis.
        """
        # Check overall performance
        if analyzed_feedback['overall_score'] < self.adaptation_threshold:
            return True

        # Check reasoning path performance
        path_analysis = analyzed_feedback.get('reasoning_analysis', {})
        if path_analysis.get('hop_effectiveness', 1.0) < 0.6:
            return True

        # Check confidence levels
        if analyzed_feedback['confidence'] < self.min_confidence:
            return True

        return False

    def get_feedback_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive summary of feedback statistics.
        """
        summary = {
            'overall_performance': self._calculate_performance_summary(),
            'reasoning_path_performance': self._calculate_path_performance(),
            'improvement_recommendations': self._generate_system_recommendations(),
            'adaptation_history': self.feedback_stats['adaptation_history']
        }

        return summary

    def _calculate_performance_summary(self) -> Dict[str, float]:
        """
        Calculate summary statistics for overall performance.
        """
        overall_scores = self.feedback_stats['performance_trends']['overall']

        if not overall_scores:
            return {'average': 0.0, 'trend': 0.0}

        return {
            'average': np.mean(overall_scores),
            'trend': self._calculate_trend(overall_scores),
            'recent_performance': np.mean(overall_scores[-10:])
        }

    def _calculate_trend(self, scores: List[float]) -> float:
        """
        Calculate the trend in a series of scores.
        """
        if len(scores) < 2:
            return 0.0

        x = np.arange(len(scores))
        y = np.array(scores)

        try:
            slope = np.polyfit(x, y, 1)[0]
            return slope
        except:
            return 0.0

    def _calculate_category_scores(self, feedback: Dict[str, Any]) -> Dict[str, float]:
        """Extract per-category numeric scores from feedback payload."""
        scores: Dict[str, float] = {}
        for category, payload in feedback.items():
            rating = payload.get('rating') if isinstance(payload, dict) else None
            if isinstance(rating, (int, float)):
                scores[category] = float(rating)
        return scores

    def _calculate_feedback_confidence(self, feedback: Dict[str, Any]) -> float:
        """
        Estimate confidence in feedback quality based on coverage + consistency.
        Returns value in [0, 1].
        """
        ratings: List[float] = []
        for payload in feedback.values():
            if isinstance(payload, dict) and isinstance(payload.get('rating'), (int, float)):
                ratings.append(float(payload['rating']))

        if not ratings:
            return 0.0

        coverage = len(ratings) / max(1, len(self.feedback_categories))
        std = float(np.std(ratings)) if len(ratings) > 1 else 0.0
        consistency = max(0.0, 1.0 - min(1.0, std / 2.0))
        return float(max(0.0, min(1.0, 0.5 * coverage + 0.5 * consistency)))

    def _analyze_performance_metrics(
            self,
            feedback: Dict[str, Any],
            metadata: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Compute lightweight runtime metrics for downstream trend analysis."""
        metadata = metadata or {}
        processing_time = metadata.get('processing_time')
        model_confidence = metadata.get('confidence')
        overall_score = self._calculate_overall_score(feedback)
        return {
            'overall_score': float(overall_score),
            'processing_time': float(processing_time) if isinstance(processing_time, (int, float)) else 0.0,
            'model_confidence': float(model_confidence) if isinstance(model_confidence, (int, float)) else 0.0
        }

    def _generate_recommendations(self, analyzed_feedback: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations from analyzed feedback."""
        recommendations: List[str] = []

        if analyzed_feedback.get('overall_score', 0.0) < self.adaptation_threshold:
            recommendations.append("Increase symbolic-neural agreement checks for low-confidence cases.")

        for area in analyzed_feedback.get('improvement_areas', []):
            category = area.get('category', 'general')
            recommendations.append(f"Improve {category} quality for similar queries.")

        for suggestion in analyzed_feedback.get('reasoning_analysis', {}).get('improvement_suggestions', []):
            recommendations.append(suggestion)

        # Keep recommendations concise and unique while preserving order
        deduped: List[str] = []
        seen = set()
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                deduped.append(rec)

        return deduped[:5]

    def _store_feedback(self, feedback_report: Dict[str, Any]) -> None:
        """Persist processed feedback report to local JSONL log."""
        output_file = self.feedback_dir / "feedback_reports.jsonl"
        with output_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_report) + "\n")

    def _calculate_path_performance(self) -> Dict[str, Dict[str, float]]:
        """Aggregate per-path feedback stats."""
        summary: Dict[str, Dict[str, float]] = {}
        for path_type, scores in self.feedback_stats['reasoning_paths'].items():
            if not scores:
                continue
            summary[path_type] = {
                'average_score': float(np.mean(scores)),
                'count': float(len(scores))
            }
        return summary

    def _generate_system_recommendations(self) -> List[str]:
        """Generate system-level recommendations from rolling trends."""
        recs: List[str] = []
        perf = self._calculate_performance_summary()
        if perf.get('average', 0.0) < self.adaptation_threshold:
            recs.append("Average feedback is below target; prioritize routing and calibration refinements.")
        if perf.get('trend', 0.0) < 0:
            recs.append("Feedback trend is declining; re-check prompt templates and fusion thresholds.")
        if not recs:
            recs.append("Feedback trends are stable; continue monitoring and periodic recalibration.")
        return recs
