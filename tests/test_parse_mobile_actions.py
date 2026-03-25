from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.data.parse_mobile_actions import (
    import_mobile_actions_dataset,
    load_mobile_actions_rows,
    parse_mobile_actions_row,
)

FIXTURE_PATH = Path("tests/fixtures/mobile_actions_smoke.jsonl")


class ParseMobileActionsTests(unittest.TestCase):
    def test_load_rows_reads_jsonl_fixture(self) -> None:
        rows = load_mobile_actions_rows(FIXTURE_PATH)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["id"], "ma-train-001")

    def test_parse_row_supports_function_tool_call_wrappers(self) -> None:
        rows = load_mobile_actions_rows(FIXTURE_PATH)
        example = parse_mobile_actions_row(rows[0], row_index=0)

        self.assertEqual(example.meta["source"], "mobile_actions")
        self.assertEqual(example.meta["source_row_id"], "ma-train-001")
        self.assertEqual(example.meta["canonical_tool_id"], "send_email")
        self.assertEqual(
            example.meta["canonical_argument_map"],
            {"recipient": "recipient", "subject": "subject"},
        )
        self.assertEqual(example.gold["arguments"]["recipient"], "sam@example.com")

    def test_parse_row_supports_explicit_no_tool_outputs(self) -> None:
        rows = load_mobile_actions_rows(FIXTURE_PATH)
        example = parse_mobile_actions_row(rows[2], row_index=2)

        self.assertEqual(example.gold, {"name": "NO_TOOL", "arguments": {}})
        self.assertIsNone(example.meta["canonical_tool_id"])
        self.assertEqual(example.meta["canonical_argument_map"], {})

    def test_parse_row_supports_metadata_split_and_uppercase_types(self) -> None:
        rows = load_mobile_actions_rows(FIXTURE_PATH)
        row = dict(rows[0])
        row.pop("split", None)
        row["raw_split"] = None
        row["metadata"] = "train"
        row["tools"][0]["function"]["parameters"]["properties"]["recipient"]["type"] = "STRING"

        example = parse_mobile_actions_row(row, row_index=0)

        self.assertEqual(example.meta["source"], "mobile_actions")
        self.assertEqual(example.tools[0].arguments[0].type, "string")

    def test_loader_supports_top_level_split_mapping_json(self) -> None:
        rows = load_mobile_actions_rows(FIXTURE_PATH)
        train_row = dict(rows[0])
        eval_row = dict(rows[-1])
        train_row.pop("split", None)
        eval_row.pop("split", None)
        payload = {"train": [train_row], "test": [eval_row]}
        fixture_path = Path("tests/_tmp/mobile_actions_split_map.json")
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        fixture_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        lifted_rows = load_mobile_actions_rows(fixture_path)

        self.assertEqual(len(lifted_rows), 2)
        self.assertEqual(lifted_rows[0]["split"], "train")
        self.assertEqual(lifted_rows[1]["split"], "test")

    def test_import_dataset_writes_manifest_and_split_manifest(self) -> None:
        output_dir = Path("tests/_tmp/mobile_actions_import")
        output_dir.mkdir(parents=True, exist_ok=True)

        result = import_mobile_actions_dataset(
            raw_path=FIXTURE_PATH,
            output_dir=output_dir,
            manifest_id="manifest_test_mobile_actions_v1",
            prompt_contract_version="fc_v1",
        )

        self.assertEqual(result.row_count, 4)
        self.assertEqual(result.canonical_manifest.example_count, 4)
        self.assertEqual(result.canonical_manifest.split_counts["eval"], 1)
        self.assertEqual(result.canonical_manifest.split_counts["val"], 1)
        self.assertEqual(result.canonical_manifest.split_counts["calib"], 1)
        self.assertEqual(result.canonical_manifest.split_counts["train"], 1)
        self.assertEqual(result.skipped_row_count, 0)
        self.assertEqual(result.no_tool_example_count, 1)
        self.assertEqual(result.unique_tool_count, 3)
        self.assertTrue(Path(result.split_manifest_path).exists())
        self.assertTrue(Path(result.summary_path).exists())

    def test_import_dataset_can_skip_out_of_scope_multi_call_rows(self) -> None:
        rows = load_mobile_actions_rows(FIXTURE_PATH)
        multi_call_row = dict(rows[0])
        multi_call_row["id"] = "ma-train-multi"
        multi_call_row["assistant"] = {
            "tool_calls": [
                {"function": {"name": "send_email", "arguments": {"recipient": "sam@example.com"}}},
                {"function": {"name": "set_alarm", "arguments": {"time": "07:30"}}},
            ]
        }
        payload_path = Path("tests/_tmp/mobile_actions_multi.jsonl")
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_text(
            json.dumps(rows[0]) + "\n" + json.dumps(multi_call_row) + "\n",
            encoding="utf-8",
        )

        output_dir = Path("tests/_tmp/mobile_actions_import_skip")
        output_dir.mkdir(parents=True, exist_ok=True)
        result = import_mobile_actions_dataset(
            raw_path=payload_path,
            output_dir=output_dir,
            manifest_id="manifest_test_mobile_actions_skip_v1",
            prompt_contract_version="fc_v1",
            skip_unsupported=True,
        )

        self.assertEqual(result.row_count, 2)
        self.assertEqual(result.retained_row_count, 1)
        self.assertEqual(result.skipped_row_count, 1)
        self.assertEqual(result.skipped_reasons["multi_tool_call_out_of_scope"], 1)


if __name__ == "__main__":
    unittest.main()
