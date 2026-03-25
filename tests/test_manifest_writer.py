from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.data.canonical import ArgSpec, ToolSpec, build_canonical_example
from src.data.manifest import build_manifest_hash, write_manifest


def _example(index: int) -> object:
    return build_canonical_example(
        source="synthetic",
        raw_split="train",
        user_request=f"Request {index}",
        tools=[
            ToolSpec(
                tool_id="send_email",
                name="send_email",
                description="Send email.",
                arguments=[
                    ArgSpec(name="recipient", type="string", required=True, description="Target"),
                ],
            )
        ],
        gold={"name": "send_email", "arguments": {"recipient": f"user-{index}@example.com"}},
        source_row_id=str(index),
    )


class ManifestWriterTests(unittest.TestCase):
    def test_manifest_hash_is_stable_for_reordered_examples(self) -> None:
        first = [_example(1), _example(2)]
        second = [_example(2), _example(1)]
        hash_a = build_manifest_hash(examples=first, prompt_contract_version="fc_v1")
        hash_b = build_manifest_hash(examples=second, prompt_contract_version="fc_v1")
        self.assertEqual(hash_a, hash_b)

    def test_write_manifest_writes_examples_and_metadata(self) -> None:
        output_dir = Path("tests/_tmp/manifest_writer")
        output_dir.mkdir(parents=True, exist_ok=True)
        record = write_manifest(
            examples=[_example(1)],
            output_dir=output_dir,
            manifest_id="manifest_test_v1",
            prompt_contract_version="fc_v1",
            metadata={"purpose": "unit-test"},
        )

        manifest_path = Path(record.manifest_path)
        dataset_path = Path(record.dataset_path)

        self.assertTrue(manifest_path.exists())
        self.assertTrue(dataset_path.exists())

        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest_payload["manifest_id"], "manifest_test_v1")
        self.assertEqual(manifest_payload["prompt_contract_version"], "fc_v1")
        self.assertEqual(manifest_payload["metadata"]["purpose"], "unit-test")


if __name__ == "__main__":
    unittest.main()
