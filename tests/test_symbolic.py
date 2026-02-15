import unittest

from src.utils.ensemble_helpers import are_drop_values_equivalent, normalize_drop_number


class TestDropHelperUtilities(unittest.TestCase):
    def test_normalize_drop_number_basic_cases(self):
        self.assertEqual(normalize_drop_number("1,234"), 1234.0)
        self.assertEqual(normalize_drop_number("three"), 3.0)
        self.assertEqual(normalize_drop_number("7.5 points"), 7.5)
        self.assertIsNone(normalize_drop_number(""))

    def test_are_drop_values_equivalent_number(self):
        pred = {"number": "10.0000"}
        gold = {"number": "10"}
        self.assertTrue(are_drop_values_equivalent(pred, gold, "number"))

    def test_are_drop_values_equivalent_number_empty_control(self):
        pred = {"number": ""}
        gold = {"number": ""}
        self.assertFalse(are_drop_values_equivalent(pred, gold, "number"))
        self.assertTrue(
            are_drop_values_equivalent(pred, gold, "number", treat_empty_as_agree=True)
        )

    def test_are_drop_values_equivalent_spans_normalized(self):
        pred = {"spans": ["The Quick, Brown Fox!"]}
        gold = {"spans": ["quick brown fox"]}
        self.assertTrue(are_drop_values_equivalent(pred, gold, "spans"))

    def test_are_drop_values_equivalent_date(self):
        pred = {"date": {"day": "7", "month": "June", "year": "2018"}}
        gold = {"date": {"day": "7", "month": "June", "year": "2018"}}
        self.assertTrue(are_drop_values_equivalent(pred, gold, "date"))
