from __future__ import annotations

import unittest

from src.data.canonical import ArgSpec, ToolSpec, build_canonical_example
from src.eval.metrics import aggregate_scores, score_prediction


def _tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            tool_id="send_email",
            name="dispatch_message",
            description="Compose and send an electronic mail message.",
            arguments=[
                ArgSpec(name="to", type="string", required=True, description="Target address"),
                ArgSpec(name="topic", type="string", required=True, description="Subject line"),
            ],
        )
    ]


class MetricsTests(unittest.TestCase):
    def test_semantic_scoring_maps_aliases_to_canonical_names(self) -> None:
        example = build_canonical_example(
            source="synthetic",
            raw_split="eval",
            user_request="Email Sam about the meeting.",
            tools=_tools(),
            gold={
                "name": "dispatch_message",
                "arguments": {"to": "sam@example.com", "topic": "meeting"},
            },
            meta={
                "canonical_tool_id": "send_email",
                "canonical_argument_map": {
                    "to": "recipient",
                    "topic": "subject",
                },
            },
        )
        score = score_prediction(
            raw_output='{"name":"dispatch_message","arguments":{"to":"sam@example.com","topic":"meeting"}}',
            example=example,
        )

        self.assertTrue(score.strict_correct)
        self.assertTrue(score.semantic_correct)
        self.assertEqual(score.semantic_predicted_name, "send_email")
        self.assertEqual(score.semantic_gold_name, "send_email")

    def test_multiple_json_objects_fail_both_strict_and_semantic(self) -> None:
        example = build_canonical_example(
            source="synthetic",
            raw_split="eval",
            user_request="Do nothing.",
            tools=[],
            gold={"name": "NO_TOOL", "arguments": {}},
        )
        score = score_prediction(
            raw_output='{"name":"NO_TOOL","arguments":{}} {"name":"NO_TOOL","arguments":{}}',
            example=example,
        )

        self.assertFalse(score.strict_correct)
        self.assertFalse(score.semantic_correct)
        self.assertEqual(score.parse_status, "multiple_objects")

    def test_nocall_detection_counts_in_aggregate_metrics(self) -> None:
        nocall_example = build_canonical_example(
            source="synthetic",
            raw_split="eval",
            user_request="Tell me a joke.",
            tools=_tools(),
            gold={"name": "NO_TOOL", "arguments": {}},
        )
        call_example = build_canonical_example(
            source="synthetic",
            raw_split="eval",
            user_request="Email Sam.",
            tools=_tools(),
            gold={
                "name": "dispatch_message",
                "arguments": {"to": "sam@example.com", "topic": "hi"},
            },
            meta={
                "canonical_tool_id": "send_email",
                "canonical_argument_map": {
                    "to": "recipient",
                    "topic": "subject",
                },
            },
        )
        scores = [
            score_prediction(
                raw_output='{"name":"NO_TOOL","arguments":{}}',
                example=nocall_example,
            ),
            score_prediction(
                raw_output='{"name":"dispatch_message","arguments":{"to":"sam@example.com","topic":"hi"}}',
                example=call_example,
            ),
        ]

        aggregate = aggregate_scores(scores)

        self.assertEqual(aggregate.total_examples, 2)
        self.assertEqual(aggregate.strict_full_call_success, 1.0)
        self.assertEqual(aggregate.semantic_full_call_success, 1.0)
        self.assertEqual(aggregate.call_no_call_f1, 1.0)

    def test_unexpected_optional_argument_fails_argument_match(self) -> None:
        example = build_canonical_example(
            source="synthetic",
            raw_split="eval",
            user_request="Email Sam.",
            tools=_tools(),
            gold={
                "name": "dispatch_message",
                "arguments": {"to": "sam@example.com", "topic": "hi"},
            },
            meta={
                "canonical_tool_id": "send_email",
                "canonical_argument_map": {
                    "to": "recipient",
                    "topic": "subject",
                },
            },
        )

        score = score_prediction(
            raw_output=(
                '{"name":"dispatch_message","arguments":'
                '{"to":"sam@example.com","topic":"hi","cc":"friend@example.com"}}'
            ),
            example=example,
        )

        self.assertFalse(score.strict_correct)
        self.assertEqual(score.strict_error, "unknown_argument")

    def test_list_arguments_do_not_crash_semantic_scoring(self) -> None:
        example = build_canonical_example(
            source="synthetic",
            raw_split="eval",
            user_request="Email Sam.",
            tools=_tools(),
            gold={
                "name": "dispatch_message",
                "arguments": {"to": "sam@example.com", "topic": "hi"},
            },
            meta={
                "canonical_tool_id": "send_email",
                "canonical_argument_map": {
                    "to": "recipient",
                    "topic": "subject",
                },
            },
        )

        score = score_prediction(
            raw_output='{"name":"dispatch_message","arguments":[]}',
            example=example,
        )

        self.assertFalse(score.strict_correct)
        self.assertFalse(score.semantic_correct)
        self.assertEqual(score.strict_error, "arguments_not_object")
        self.assertEqual(score.semantic_error, "arguments_not_object")


if __name__ == "__main__":
    unittest.main()
