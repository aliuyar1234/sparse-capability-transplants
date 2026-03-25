from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

from src.data.canonical import ArgSpec, ToolSpec, build_canonical_example
from src.data.freeze_eval_artifacts import run_eval_freeze_pipeline


def _canonical_examples() -> list:
    send_email = ToolSpec(
        tool_id="send_email",
        name="send_email",
        description="Sends an email.",
        arguments=[
            ArgSpec(name="to", type="string", required=True, description="Email recipient."),
            ArgSpec(name="subject", type="string", required=True, description="Email subject."),
        ],
    )
    show_map = ToolSpec(
        tool_id="show_map",
        name="show_map",
        description="Shows a location on the map.",
        arguments=[
            ArgSpec(name="query", type="string", required=True, description="Place or address."),
        ],
    )
    train_example = build_canonical_example(
        source="mobile_actions",
        raw_split="train",
        user_request="Email Sam about the report.",
        tools=[send_email, show_map],
        gold={"name": "send_email", "arguments": {"to": "sam@example.com", "subject": "report"}},
        source_row_id="row-train-1",
        meta={
            "source_row_id": "row-train-1",
            "canonical_tool_id": "send_email",
            "canonical_argument_map": {"to": "to", "subject": "subject"},
            "alias_bank_id": "none",
        },
    )
    eval_example = build_canonical_example(
        source="mobile_actions",
        raw_split="eval",
        user_request="Show me the cafe on Market Street.",
        tools=[send_email, show_map],
        gold={"name": "show_map", "arguments": {"query": "cafe on Market Street"}},
        source_row_id="row-eval-1",
        meta={
            "source_row_id": "row-eval-1",
            "canonical_tool_id": "show_map",
            "canonical_argument_map": {"query": "query"},
            "alias_bank_id": "none",
        },
    )
    return [
        replace(train_example, split="train"),
        replace(eval_example, split="eval"),
    ]


class FreezeEvalArtifactsTests(unittest.TestCase):
    def test_eval_freeze_pipeline_writes_real_eval_artifacts(self) -> None:
        output_dir = Path("tests/_tmp/freeze_eval_artifacts")
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = run_eval_freeze_pipeline(
            canonical_examples=_canonical_examples(),
            canonical_manifest_payload={
                "manifest_id": "manifest_fixture_canonical_v1",
                "manifest_hash": "fixturehash001",
                "prompt_contract_version": "fc_v1",
                "metadata": {"split_manifest_hash": "splitfixture001"},
            },
            output_dir=output_dir,
        )

        self.assertEqual(summary["counts"]["iid_examples"], 1)
        self.assertEqual(summary["counts"]["schema_shift_examples"], 1)
        self.assertEqual(summary["counts"]["distractor_examples"], 1)
        self.assertEqual(summary["counts"]["nocall_missing_examples"], 1)
        self.assertEqual(summary["counts"]["nocall_unsupported_examples"], 1)
        self.assertEqual(summary["counts"]["control_examples"], 1)
        self.assertEqual(summary["evaluation_manifest"]["split_counts"]["eval"], 5)

        leakage_payload = json.loads(
            Path(summary["leakage_audit_path"]).read_text(encoding="utf-8")
        )
        self.assertEqual(leakage_payload["summary"]["alias_bank_collision_count"], 0)
        self.assertEqual(leakage_payload["summary"]["derived_non_eval_source_count"], 0)

        golden_payload = json.loads(
            Path(summary["golden_fixture_path"]).read_text(encoding="utf-8")
        )
        self.assertEqual(golden_payload["prompt_contract_version"], "fc_v1")


if __name__ == "__main__":
    unittest.main()
