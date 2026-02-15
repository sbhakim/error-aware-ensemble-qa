import unittest
import tempfile
from pathlib import Path

try:
    from src.system.ensemble_manager import EnsembleManager
    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-dependent
    EnsembleManager = None
    IMPORT_ERROR = exc


@unittest.skipIf(EnsembleManager is None, f"ensemble manager import failed: {IMPORT_ERROR}")
class TestEnsembleManagerHelpers(unittest.TestCase):
    @staticmethod
    def _new_manager(config):
        manager = EnsembleManager.__new__(EnsembleManager)
        manager.config = config
        return manager

    def test_resolve_rules_file_prefers_model_override_for_drop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            override = Path(tmpdir) / "override_rules.json"
            dynamic = Path(tmpdir) / "dynamic_rules.json"
            static = Path(tmpdir) / "static_rules.json"
            override.write_text("{}", encoding="utf-8")
            dynamic.write_text("{}", encoding="utf-8")
            static.write_text("{}", encoding="utf-8")

            manager = self._new_manager({
                "drop_rules_dynamic_file": str(dynamic),
                "drop_rules_file": str(static)
            })
            chosen = manager._resolve_rules_file(
                model_cfg={"rules_file": str(override)},
                dataset_type="drop"
            )
            self.assertEqual(chosen, str(override))

    def test_resolve_rules_file_drop_fallback_dynamic_then_static(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dynamic = Path(tmpdir) / "dynamic_rules.json"
            static = Path(tmpdir) / "static_rules.json"
            dynamic.write_text("{}", encoding="utf-8")
            static.write_text("{}", encoding="utf-8")

            manager = self._new_manager({
                "drop_rules_dynamic_file": str(dynamic),
                "drop_rules_file": str(static)
            })

            self.assertEqual(
                manager._resolve_rules_file(model_cfg={}, dataset_type="drop"),
                str(dynamic)
            )

            dynamic.unlink()
            self.assertEqual(
                manager._resolve_rules_file(model_cfg={}, dataset_type="drop"),
                str(static)
            )

    def test_extract_response_details_handles_tuple_payload(self):
        manager = self._new_manager({})
        response = (
            {"answer": {"number": "5", "spans": [], "date": {"day": "", "month": "", "year": ""}, "confidence": 0.8}},
            "hybrid"
        )
        extracted = manager._extract_response_details(response)
        self.assertEqual(extracted["reasoning_path"], "hybrid")
        self.assertEqual(extracted["answer"]["number"], "5")
        self.assertAlmostEqual(extracted["confidence"], 0.8, places=6)
