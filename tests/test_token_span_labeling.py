from __future__ import annotations

import unittest

import torch

from src.train.cache_activations import (
    TOKEN_CLASS_ARGKEY,
    TOKEN_CLASS_ARGVAL,
    TOKEN_CLASS_DECISION,
    TOKEN_CLASS_OTHER,
    TOKEN_CLASS_TOOL,
    build_target_char_spans,
    label_output_token_classes,
    selected_token_positions,
)


class TokenSpanLabelingTests(unittest.TestCase):
    def test_build_target_char_spans_matches_locked_json_layout(self) -> None:
        target = {
            "name": "dispatch_message",
            "arguments": {
                "subject": "Quarterly Sales Figures",
                "to": "sam@example.com",
            },
        }
        spans = build_target_char_spans(target)
        labels = [span.token_class for span in spans]

        self.assertEqual(labels.count(TOKEN_CLASS_DECISION), 1)
        self.assertEqual(labels.count(TOKEN_CLASS_TOOL), 1)
        self.assertEqual(labels.count(TOKEN_CLASS_ARGKEY), 2)
        self.assertEqual(labels.count(TOKEN_CLASS_ARGVAL), 2)

    def test_label_output_token_classes_maps_char_spans_to_output_tokens(self) -> None:
        prompt_text = "PROMPT:"
        offsets = [
            (0, len(prompt_text)),
            (len(prompt_text) + 0, len(prompt_text) + 1),
            (len(prompt_text) + 1, len(prompt_text) + 7),
            (len(prompt_text) + 8, len(prompt_text) + 26),
            (len(prompt_text) + 26, len(prompt_text) + 39),
            (len(prompt_text) + 40, len(prompt_text) + 44),
            (len(prompt_text) + 45, len(prompt_text) + 62),
            (len(prompt_text) + 62, len(prompt_text) + 64),
        ]

        labels = label_output_token_classes(
            offset_mapping=offsets,
            output_start_char=len(prompt_text),
            target={"name": "dispatch_message", "arguments": {"to": "sam@example.com"}},
        )

        self.assertEqual(
            labels,
            [
                None,
                TOKEN_CLASS_DECISION,
                TOKEN_CLASS_OTHER,
                TOKEN_CLASS_TOOL,
                TOKEN_CLASS_OTHER,
                TOKEN_CLASS_ARGKEY,
                TOKEN_CLASS_ARGVAL,
                TOKEN_CLASS_OTHER,
            ],
        )
        self.assertEqual(
            selected_token_positions(labels, [TOKEN_CLASS_TOOL, TOKEN_CLASS_ARGVAL]),
            [3, 6],
        )

    def test_offset_tensor_rows_are_supported(self) -> None:
        labels = label_output_token_classes(
            offset_mapping=[tuple(pair.tolist()) for pair in torch.tensor([[2, 3], [3, 5]])],
            output_start_char=2,
            target={"name": "x", "arguments": {}},
        )
        self.assertEqual(labels[0], TOKEN_CLASS_DECISION)


if __name__ == "__main__":
    unittest.main()
