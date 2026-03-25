from __future__ import annotations

from pathlib import Path
from typing import Any

from src.analysis.paper_artifacts import load_json, write_csv, write_json


def _current_claim_statuses(claims_matrix_path: str | Path) -> dict[str, str]:
    statuses: dict[str, str] = {}
    current_claim_id: str | None = None
    expect_status_line_for: str | None = None
    for line in Path(claims_matrix_path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("## C"):
            parts = stripped.split()
            current_claim_id = parts[1] if len(parts) >= 2 else None
            expect_status_line_for = None
        elif current_claim_id and stripped.startswith("- **Status**"):
            expect_status_line_for = current_claim_id
        elif expect_status_line_for and stripped and not stripped.startswith("- "):
            statuses[expect_status_line_for] = stripped
            expect_status_line_for = None
            current_claim_id = None
    return statuses


def _final_claim_rows(
    *,
    current_statuses: dict[str, str],
    dense_multiseed_path: str | Path,
    prune_summary_path: str | Path,
    error_summary_path: str | Path,
) -> list[dict[str, Any]]:
    return [
        {
            "claim_id": "C1",
            "old_status": current_statuses.get("C1", "unknown"),
            "new_status": "weakened",
            "bar_assessment": "failure",
            "reason": (
                "Sparse same-size transfer stayed positive across seeds, but the matched dense "
                "multiseed control outperformed sparse on the primary metric."
            ),
            "draft_language": (
                "Same-size within-family transfer is real on this task, but the sparse method "
                "does not beat the matched dense shortcut on the final multiseed comparison."
            ),
            "supporting_artifact": str(Path(dense_multiseed_path).resolve()),
        },
        {
            "claim_id": "C3",
            "old_status": current_statuses.get("C3", "unknown"),
            "new_status": "weakened",
            "bar_assessment": "weak_support",
            "reason": (
                "Primary semantic metrics improved, but schema-shift strict success remained "
                "near base level and the strongest gains still concentrate in the NoCall slices."
            ),
            "draft_language": (
                "The method improves structured function-calling behavior on the locked primary "
                "task, but the present evidence is not strong enough for a clean semantic-transfer "
                "claim."
            ),
            "supporting_artifact": str(Path(error_summary_path).resolve()),
        },
        {
            "claim_id": "C5",
            "old_status": current_statuses.get("C5", "unknown"),
            "new_status": "weakened",
            "bar_assessment": "weak_support",
            "reason": (
                "Control drops stayed within the predeclared tolerance, but sparse was not clearly "
                "less damaging than the dense shortcut on the matched multiseed comparison."
            ),
            "draft_language": (
                "Collateral damage stayed numerically low on the frozen control suite, but the "
                "paper should report the control drops directly instead of claiming a stronger "
                "narrowness win."
            ),
            "supporting_artifact": str(Path(dense_multiseed_path).resolve()),
        },
        {
            "claim_id": "C6",
            "old_status": current_statuses.get("C6", "unknown"),
            "new_status": "weakened",
            "bar_assessment": "weak_support",
            "reason": (
                "A locked 1-feature subset beat random 1-feature controls, but it retained only "
                "47.1% of the full sparse gain rather than the >=80% support bar."
            ),
            "draft_language": (
                "A small selected subset retains much of the observed sparse effect on this task, "
                "without implying uniqueness or full identifiability."
            ),
            "supporting_artifact": str(Path(prune_summary_path).resolve()),
        },
        {
            "claim_id": "C7",
            "old_status": current_statuses.get("C7", "unknown"),
            "new_status": "weakened",
            "bar_assessment": "weak_support",
            "reason": (
                "Steering and random subset controls were weaker, but the matched dense shortcut "
                "did not fail; it exceeded the sparse multiseed result."
            ),
            "draft_language": (
                "Some shortcut controls fail to explain the result, but dense parameter-matched "
                "adapters remain competitive or stronger, so the paper should not generalize "
                "beyond those ablations."
            ),
            "supporting_artifact": str(Path(dense_multiseed_path).resolve()),
        },
        {
            "claim_id": "C8",
            "old_status": current_statuses.get("C8", "supported by design"),
            "new_status": "supported",
            "bar_assessment": "support",
            "reason": (
                "The final draft remains V24-only and makes no cross-scale or universal-transfer "
                "claim."
            ),
            "draft_language": (
                "This is a same-size, within-family, single-turn function-calling study with no "
                "claim to universal transfer, general model editing, or safety."
            ),
            "supporting_artifact": "docs/RESEARCH_BRIEF.md",
        },
    ]


def _artifact_inventory(
    *,
    paper_config: dict[str, Any],
    error_summary_path: str | Path,
    recovery_summary_path: str | Path,
    tradeoff_summary_path: str | Path,
    final_claim_audit_path: str | Path,
) -> list[dict[str, Any]]:
    return [
        {
            "artifact_id": "m1_eval_freeze",
            "category": "data",
            "milestone": "M1",
            "path": str(Path(paper_config["eval_manifest_path"]).resolve()),
            "role": "Locked evaluation manifest and prompt contract source.",
            "claim_ids": "C3,C5,C7,C8",
        },
        {
            "artifact_id": "m3_baseline_summary",
            "category": "baseline",
            "milestone": "M3",
            "path": str(Path(paper_config["baseline_summary_path"]).resolve()),
            "role": "Base/donor metrics and donor-gap denominator.",
            "claim_ids": "C1,C3",
        },
        {
            "artifact_id": "m5_same_size_selected_eval",
            "category": "main_result",
            "milestone": "M5",
            "path": str(Path(paper_config["sparse_selected_eval_summary_path"]).resolve()),
            "role": "Selected sparse same-size frozen-manifest evaluation.",
            "claim_ids": "C1,C3,C5",
        },
        {
            "artifact_id": "m5_pruning",
            "category": "ablation",
            "milestone": "M5",
            "path": str(Path(paper_config["prune_summary_path"]).resolve()),
            "role": "Subset retention and random-subset controls.",
            "claim_ids": "C6,C7",
        },
        {
            "artifact_id": "m5_shortcuts",
            "category": "ablation",
            "milestone": "M5",
            "path": str(Path(paper_config["shortcut_summary_path"]).resolve()),
            "role": "Dense and steering shortcut controls.",
            "claim_ids": "C1,C5,C7",
        },
        {
            "artifact_id": "m5_sparse_multiseed",
            "category": "confirmatory",
            "milestone": "M5",
            "path": str(Path(paper_config["sparse_multiseed_summary_path"]).resolve()),
            "role": "Sparse confirmatory multi-seed aggregate.",
            "claim_ids": "C1,C5,C7",
        },
        {
            "artifact_id": "m5_dense_multiseed",
            "category": "confirmatory",
            "milestone": "M5",
            "path": str(Path(paper_config["dense_multiseed_summary_path"]).resolve()),
            "role": "Matched dense multiseed comparator.",
            "claim_ids": "C1,C5,C7",
        },
        {
            "artifact_id": "m7_error_analysis",
            "category": "analysis",
            "milestone": "M7",
            "path": str(Path(error_summary_path).resolve()),
            "role": "Strict-vs-semantic table, error categories, appendix examples.",
            "claim_ids": "C3,C5",
        },
        {
            "artifact_id": "m7_recovery_plots",
            "category": "analysis",
            "milestone": "M7",
            "path": str(Path(recovery_summary_path).resolve()),
            "role": "Baseline table, parameter budget table, retained-gain figure.",
            "claim_ids": "C1,C6,C7",
        },
        {
            "artifact_id": "m7_tradeoff_plots",
            "category": "analysis",
            "milestone": "M7",
            "path": str(Path(tradeoff_summary_path).resolve()),
            "role": (
                "Control tradeoff, calibration sensitivity, per-slice comparison tables/figures."
            ),
            "claim_ids": "C1,C3,C5,C7",
        },
        {
            "artifact_id": "m8_claim_audit",
            "category": "packaging",
            "milestone": "M8",
            "path": str(Path(final_claim_audit_path).resolve()),
            "role": "Final claim-support summary and weakened-language guardrails.",
            "claim_ids": "C1,C3,C5,C6,C7,C8",
        },
    ]


def _section_map() -> dict[str, list[str]]:
    return {
        "Introduction": ["m3_baseline_summary", "m5_dense_multiseed"],
        "Problem Setup and Scope": ["m1_eval_freeze", "m8_claim_audit"],
        "Method": ["m5_same_size_selected_eval", "m5_pruning", "m5_shortcuts"],
        "Evaluation Protocol": ["m7_error_analysis", "m7_tradeoff_plots"],
        "Baselines and Controls": ["m3_baseline_summary", "m5_shortcuts", "m5_dense_multiseed"],
        "Main Results": ["m5_same_size_selected_eval", "m5_sparse_multiseed", "m5_dense_multiseed"],
        "Mechanistic Analysis": ["m5_pruning", "m7_recovery_plots", "m7_tradeoff_plots"],
        "Robustness and Failure Modes": ["m7_error_analysis", "m8_claim_audit"],
        "Limitations": ["m8_claim_audit"],
    }


def export_final_registry(*, config: dict[str, Any], output_dir: str | Path) -> Path:
    paper_config = dict(config.get("paper_artifacts", {}))
    final_registry_config = dict(config.get("final_registry", {}))
    for key in (
        "baseline_summary_path",
        "prune_summary_path",
        "shortcut_summary_path",
        "sparse_multiseed_summary_path",
        "dense_multiseed_summary_path",
    ):
        if not paper_config.get(key):
            raise ValueError(f"paper_artifacts.{key} is required.")
    for key in (
        "error_analysis_summary_path",
        "plot_recovery_summary_path",
        "plot_tradeoffs_summary_path",
    ):
        if not final_registry_config.get(key):
            raise ValueError(f"final_registry.{key} is required.")

    claims_matrix_path = str(
        final_registry_config.get("claims_matrix_path", "docs/CLAIMS_MATRIX.md")
    )
    current_statuses = _current_claim_statuses(claims_matrix_path)

    load_json(paper_config["sparse_multiseed_summary_path"])
    load_json(paper_config["dense_multiseed_summary_path"])
    load_json(paper_config["prune_summary_path"])
    load_json(paper_config["shortcut_summary_path"])
    load_json(final_registry_config["error_analysis_summary_path"])
    load_json(final_registry_config["plot_recovery_summary_path"])
    load_json(final_registry_config["plot_tradeoffs_summary_path"])

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    m5_decision = {
        "status": "failure",
        "reason": (
            "The matched dense multiseed control exceeded the sparse multiseed result, so the "
            "strong sparse-wins same-size headline fails the final M5 bar."
        ),
        "weakened_paper_path": (
            "Write the paper as same-size transfer plus partial sparse localization, with a "
            "dense-vs-sparse performance/interpretability tradeoff rather than a "
            "sparse-superiority claim."
        ),
    }

    claim_rows = _final_claim_rows(
        current_statuses=current_statuses,
        dense_multiseed_path=paper_config["dense_multiseed_summary_path"],
        prune_summary_path=paper_config["prune_summary_path"],
        error_summary_path=final_registry_config["error_analysis_summary_path"],
    )
    final_claim_audit_path = write_json(
        destination / "final_claim_audit.json",
        {
            "status": "passed",
            "execution_variant": str(config["execution_variant"]),
            "m5_decision": m5_decision,
            "claims": claim_rows,
        },
    )

    inventory_rows = _artifact_inventory(
        paper_config=paper_config,
        error_summary_path=final_registry_config["error_analysis_summary_path"],
        recovery_summary_path=final_registry_config["plot_recovery_summary_path"],
        tradeoff_summary_path=final_registry_config["plot_tradeoffs_summary_path"],
        final_claim_audit_path=final_claim_audit_path,
    )
    inventory_json = write_json(destination / "artifact_inventory.json", {"rows": inventory_rows})
    inventory_csv = write_csv(destination / "artifact_inventory.csv", inventory_rows)
    section_map_path = write_json(destination / "section_to_artifact_map.json", _section_map())

    checklist_lines = [
        "# Final Claim Audit Checklist",
        "",
        f"- Execution variant: `{config['execution_variant']}`",
        f"- M5 decision: `{m5_decision['status']}`",
        "",
        "## Claim outcomes",
    ]
    for row in claim_rows:
        checklist_lines.append(
            f"- `{row['claim_id']}`: `{row['new_status']}`; {row['draft_language']}"
        )
    checklist_path = destination / "claim_audit_checklist.md"
    checklist_path.write_text("\n".join(checklist_lines) + "\n", encoding="utf-8")

    registry = {
        "status": "passed",
        "execution_variant": str(config["execution_variant"]),
        "m5_decision": m5_decision,
        "error_analysis_summary_path": str(
            Path(final_registry_config["error_analysis_summary_path"]).resolve()
        ),
        "plot_recovery_summary_path": str(
            Path(final_registry_config["plot_recovery_summary_path"]).resolve()
        ),
        "plot_tradeoffs_summary_path": str(
            Path(final_registry_config["plot_tradeoffs_summary_path"]).resolve()
        ),
        "artifact_inventory_path": str(inventory_json.resolve()),
        "artifact_inventory_csv_path": str(inventory_csv.resolve()),
        "section_to_artifact_map_path": str(section_map_path.resolve()),
        "final_claim_audit_path": str(final_claim_audit_path.resolve()),
        "claim_audit_checklist_path": str(checklist_path.resolve()),
        "paper_direction": (
            "Same-size transfer is real, dense parameter-matched controls outperform sparse on "
            "the matched multiseed comparison, and sparse retains value mainly as a "
            "localization/interpretability signal."
        ),
        "source_summaries": {
            "sparse_multiseed_summary_path": str(
                Path(paper_config["sparse_multiseed_summary_path"]).resolve()
            ),
            "dense_multiseed_summary_path": str(
                Path(paper_config["dense_multiseed_summary_path"]).resolve()
            ),
            "error_analysis_summary_path": str(
                Path(final_registry_config["error_analysis_summary_path"]).resolve()
            ),
            "plot_recovery_summary_path": str(
                Path(final_registry_config["plot_recovery_summary_path"]).resolve()
            ),
            "plot_tradeoffs_summary_path": str(
                Path(final_registry_config["plot_tradeoffs_summary_path"]).resolve()
            ),
        },
    }
    return write_json(destination / "summary.json", registry)
