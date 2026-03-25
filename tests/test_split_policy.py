from __future__ import annotations

import unittest

from src.data.canonical import ArgSpec, ToolSpec, build_canonical_example
from src.data.splits import assign_locked_splits


def _tool() -> ToolSpec:
    return ToolSpec(
        tool_id="send_email",
        name="send_email",
        description="Send an email.",
        arguments=[
            ArgSpec(name="recipient", type="string", required=True, description="Target address"),
        ],
    )


def _example(index: int, raw_split: str) -> object:
    return build_canonical_example(
        source="mobile_actions",
        raw_split=raw_split,
        user_request=f"Request {index}",
        tools=[_tool()],
        gold={"name": "send_email", "arguments": {"recipient": f"user-{index}@example.com"}},
        source_row_id=str(index),
    )


class SplitPolicyTests(unittest.TestCase):
    def test_assign_locked_splits_uses_fallback_percentages_for_small_train_pools(self) -> None:
        examples = [_example(index, "train") for index in range(100)]
        assigned, manifest = assign_locked_splits(examples)

        split_counts: dict[str, int] = {}
        for example in assigned:
            split_counts[example.split] = split_counts.get(example.split, 0) + 1

        self.assertEqual(split_counts["val"], 10)
        self.assertEqual(split_counts["calib"], 15)
        self.assertEqual(split_counts["train"], 75)
        self.assertEqual(manifest.counts["eval"], 0)

    def test_assign_locked_splits_preserves_eval_rows(self) -> None:
        examples = [_example(index, "eval") for index in range(3)] + [
            _example(index + 10, "train") for index in range(20)
        ]
        assigned, manifest = assign_locked_splits(examples)
        eval_rows = [example for example in assigned if example.split == "eval"]

        self.assertEqual(len(eval_rows), 3)
        self.assertEqual(manifest.counts["eval"], 3)


if __name__ == "__main__":
    unittest.main()
