import unittest
import json
import tempfile
import importlib.util
from pathlib import Path

_FEEDBACK_HANDLER_PATH = Path(__file__).resolve().parents[1] / "src" / "feedback" / "feedback_handler.py"
_spec = importlib.util.spec_from_file_location("feedback_handler_module", _FEEDBACK_HANDLER_PATH)
_module = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_module)
FeedbackHandler = _module.FeedbackHandler


class DummyFeedbackManager:
    pass


class TestFeedbackHandler(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.handler = FeedbackHandler(
            feedback_manager=DummyFeedbackManager(),
            feedback_dir=self._tmpdir.name,
            interactive=False
        )

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_non_interactive_rating_defaults_to_neutral(self):
        self.assertEqual(self.handler._get_validated_rating("ignored"), 3)

    def test_analyze_feedback_contains_core_fields(self):
        feedback = {
            "accuracy": {"rating": 2, "comment": "wrong value"},
            "reasoning": {"rating": 3, "comment": "unclear chain"},
            "completeness": {"rating": 4, "comment": None}
        }
        reasoning_path = {"type": "hybrid", "hop_count": 2, "confidence": 0.55}
        metadata = {"processing_time": 1.2, "confidence": 0.61}

        analysis = self.handler._analyze_feedback(feedback, reasoning_path, metadata)

        self.assertIn("overall_score", analysis)
        self.assertIn("category_scores", analysis)
        self.assertIn("confidence", analysis)
        self.assertIn("performance_metrics", analysis)
        self.assertIn("reasoning_analysis", analysis)
        self.assertGreaterEqual(analysis["confidence"], 0.0)
        self.assertLessEqual(analysis["confidence"], 1.0)

    def test_store_feedback_writes_jsonl_row(self):
        report = {"timestamp": "2026-01-01T00:00:00", "status": "ok"}
        self.handler._store_feedback(report)

        output_file = Path(self._tmpdir.name) / "feedback_reports.jsonl"
        self.assertTrue(output_file.exists())

        lines = output_file.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["status"], "ok")
