from __future__ import annotations

import unittest

from src.data.canonical import ArgSpec
from src.eval.normalize_args import normalize_argument_value


class NormalizeArgsTests(unittest.TestCase):
    def test_string_normalization_trims_and_lowercases(self) -> None:
        result = normalize_argument_value(
            "  Sam@Example.com  ", ArgSpec("recipient", "string", True, "")
        )
        self.assertIsNone(result.error)
        self.assertEqual(result.value, "sam@example.com")

    def test_phone_normalizer_strips_punctuation(self) -> None:
        result = normalize_argument_value(
            "(555) 123-4567",
            {
                "name": "phone",
                "type": "string",
                "required": True,
                "description": "",
                "normalizer": "phone",
            },
        )
        self.assertIsNone(result.error)
        self.assertEqual(result.value, "5551234567")

    def test_bool_normalizer_is_deterministic(self) -> None:
        result = normalize_argument_value("YES", ArgSpec("notify", "bool", False, ""))
        self.assertIsNone(result.error)
        self.assertTrue(result.value)

    def test_invalid_bool_reports_error(self) -> None:
        result = normalize_argument_value("sometimes", ArgSpec("notify", "bool", False, ""))
        self.assertEqual(result.error, "invalid_bool")


if __name__ == "__main__":
    unittest.main()
