from __future__ import annotations

import unittest

from src.eval.parse_json import extract_first_valid_json_object


class ParseJsonTests(unittest.TestCase):
    def test_extracts_json_after_prose(self) -> None:
        result = extract_first_valid_json_object(
            'Sure: {"name":"send_email","arguments":{"to":"sam"}}'
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.parsed, {"name": "send_email", "arguments": {"to": "sam"}})

    def test_skips_malformed_candidate_and_finds_later_valid_object(self) -> None:
        result = extract_first_valid_json_object(
            '{"name": } then {"name":"NO_TOOL","arguments":{}}'
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.parsed, {"name": "NO_TOOL", "arguments": {}})

    def test_flags_multiple_valid_objects(self) -> None:
        result = extract_first_valid_json_object('{"name":"A"} {"name":"B"}')
        self.assertEqual(result.status, "multiple_objects")
        self.assertEqual(result.valid_object_count, 2)

    def test_ignores_braces_inside_strings(self) -> None:
        result = extract_first_valid_json_object(
            '{"name":"send_email","arguments":{"body":"Use {braces} literally"}}'
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.valid_object_count, 1)


if __name__ == "__main__":
    unittest.main()
