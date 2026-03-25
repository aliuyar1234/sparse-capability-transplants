from __future__ import annotations

from dataclasses import dataclass

from src.data.build_control_suite import ControlExample


@dataclass(frozen=True)
class ControlScore:
    example_id: str
    exact_match: bool
    normalized_prediction: str
    normalized_target: str


def normalize_control_text(text: str) -> str:
    return text.replace("\r\n", "\n").strip()


def score_control_prediction(
    *,
    raw_output: str,
    example: ControlExample,
) -> ControlScore:
    normalized_prediction = normalize_control_text(raw_output)
    normalized_target = normalize_control_text(example.target_text)
    return ControlScore(
        example_id=example.example_id,
        exact_match=normalized_prediction == normalized_target,
        normalized_prediction=normalized_prediction,
        normalized_target=normalized_target,
    )


def aggregate_control_scores(scores: list[ControlScore]) -> float:
    if not scores:
        return 0.0
    return sum(score.exact_match for score in scores) / len(scores)
