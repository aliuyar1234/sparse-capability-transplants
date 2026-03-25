from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys


def _run(command: list[str]) -> int:
    try:
        completed = subprocess.run(command, check=False)
    except KeyboardInterrupt:
        return 130
    return completed.returncode


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-platform repo task runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("env", help="Install the project with development dependencies.")
    subparsers.add_parser("lint", help="Run ruff checks.")
    subparsers.add_parser("test", help="Run the test suite.")

    smoke_data = subparsers.add_parser(
        "smoke-data",
        help="Run the synthetic data harness smoke test.",
    )
    smoke_data.add_argument("--config", required=True)

    import_mobile_actions = subparsers.add_parser(
        "import-mobile-actions",
        help="Import Mobile Actions-style raw rows into canonical manifests.",
    )
    import_mobile_actions.add_argument("--config", required=True)

    train_donor = subparsers.add_parser(
        "train-donor",
        help="Run the donor smoke-preflight training pipeline.",
    )
    train_donor.add_argument("--config", required=True)

    train_recipient = subparsers.add_parser(
        "train-recipient-baselines",
        help="Run a recipient baseline smoke training pipeline.",
    )
    train_recipient.add_argument("--config", required=True)

    cache_activations = subparsers.add_parser(
        "cache-activations",
        help="Collect teacher-forced donor/base activation caches.",
    )
    cache_activations.add_argument("--config", required=True)

    rank_layers = subparsers.add_parser(
        "rank-layers",
        help="Fit rough sparse modules and rank candidate layers from cached activations.",
    )
    rank_layers.add_argument("--config", required=True)

    eval_layer_candidate = subparsers.add_parser(
        "eval-layer-candidate",
        help="Run end-to-end single-layer rough-candidate scoring.",
    )
    eval_layer_candidate.add_argument("--config", required=True)

    eval_main = subparsers.add_parser(
        "eval-main",
        help="Run the deterministic evaluation wrapper.",
    )
    eval_main.add_argument("--config", required=True)

    fit_same_size = subparsers.add_parser(
        "fit-same-size-transplant",
        help="Run the first M5 same-size fitting slice.",
    )
    fit_same_size.add_argument("--config", required=True)

    prune_features = subparsers.add_parser(
        "prune-features",
        help="Run the M5 feature-pruning slice.",
    )
    prune_features.add_argument("--config", required=True)

    shortcut_controls = subparsers.add_parser(
        "same-size-shortcut-controls",
        help="Run the M5 same-size shortcut-control slice.",
    )
    shortcut_controls.add_argument("--config", required=True)

    same_size_multiseed = subparsers.add_parser(
        "same-size-multiseed",
        help="Run the V24-S6 same-size confirmatory multi-seed slice.",
    )
    same_size_multiseed.add_argument("--config", required=True)

    dense_control_multiseed = subparsers.add_parser(
        "same-size-dense-multiseed-control",
        help="Run the bounded M5 dense shortcut multi-seed follow-up.",
    )
    dense_control_multiseed.add_argument("--config", required=True)

    error_analysis = subparsers.add_parser(
        "error-analysis",
        help="Generate the M7 strict-vs-semantic and error-category artifacts.",
    )
    error_analysis.add_argument("--config", required=True)

    plot_recovery = subparsers.add_parser(
        "plot-recovery",
        help="Generate the M7 recovery and retained-gain tables/figures.",
    )
    plot_recovery.add_argument("--config", required=True)

    plot_tradeoffs = subparsers.add_parser(
        "plot-tradeoffs",
        help="Generate the M7 same-size vs control tradeoff artifacts.",
    )
    plot_tradeoffs.add_argument("--config", required=True)

    export_registry = subparsers.add_parser(
        "export-final-registry",
        help="Generate the M8 final claim audit and artifact registry.",
    )
    export_registry.add_argument("--config", required=True)

    param_budget = subparsers.add_parser(
        "param-budget",
        help="Write a parameter-budget smoke artifact.",
    )
    param_budget.add_argument("--config", required=True)

    summarize_baselines = subparsers.add_parser(
        "summarize-baselines",
        help="Summarize donor/base prediction artifacts and compute the donor-gap metric.",
    )
    summarize_baselines.add_argument("--config", required=True)

    donor_gap_gate = subparsers.add_parser(
        "donor-gap-gate",
        help="Write the explicit R20 donor-gap gate artifact.",
    )
    donor_gap_gate.add_argument("--config", required=True)

    freeze_mobile_actions_eval = subparsers.add_parser(
        "freeze-mobile-actions-eval",
        help="Freeze real-corpus M1 evaluation artifacts from an imported canonical manifest.",
    )
    freeze_mobile_actions_eval.add_argument("--config", required=True)

    smoke_model = subparsers.add_parser("smoke-model", help="Run the model smoke probe.")
    smoke_model.add_argument("--config", required=True)

    manifest_smoke = subparsers.add_parser(
        "manifest-smoke",
        help="Write a manifest smoke artifact.",
    )
    manifest_smoke.add_argument("--config", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "env":
        return _run([sys.executable, "-m", "pip", "install", "-e", ".[dev]"])

    if args.command == "lint":
        if not _has_module("ruff"):
            print("ruff is not installed. Run `python scripts/task.py env` first.", file=sys.stderr)
            return 1
        commands = [
            [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"],
            [sys.executable, "-m", "ruff", "format", "--check", "src", "tests", "scripts"],
        ]
        exit_codes = [_run(command) for command in commands]
        return max(exit_codes)

    if args.command == "test":
        if _has_module("pytest"):
            return _run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"])
        return _run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])

    if args.command == "smoke-model":
        return _run([sys.executable, "-m", "src.cli", "smoke-model", "--config", args.config])

    if args.command == "smoke-data":
        return _run([sys.executable, "-m", "src.cli", "smoke-data", "--config", args.config])

    if args.command == "import-mobile-actions":
        return _run(
            [sys.executable, "-m", "src.cli", "import-mobile-actions", "--config", args.config]
        )

    if args.command == "train-donor":
        return _run([sys.executable, "-m", "src.cli", "train-donor", "--config", args.config])

    if args.command == "train-recipient-baselines":
        return _run(
            [sys.executable, "-m", "src.cli", "train-recipient-baselines", "--config", args.config]
        )

    if args.command == "cache-activations":
        return _run([sys.executable, "-m", "src.cli", "cache-activations", "--config", args.config])

    if args.command == "rank-layers":
        return _run([sys.executable, "-m", "src.cli", "rank-layers", "--config", args.config])

    if args.command == "eval-layer-candidate":
        return _run(
            [sys.executable, "-m", "src.cli", "eval-layer-candidate", "--config", args.config]
        )

    if args.command == "eval-main":
        return _run([sys.executable, "-m", "src.cli", "eval-main", "--config", args.config])

    if args.command == "fit-same-size-transplant":
        return _run(
            [sys.executable, "-m", "src.cli", "fit-same-size-transplant", "--config", args.config]
        )

    if args.command == "prune-features":
        return _run([sys.executable, "-m", "src.cli", "prune-features", "--config", args.config])

    if args.command == "same-size-shortcut-controls":
        return _run(
            [
                sys.executable,
                "-m",
                "src.cli",
                "same-size-shortcut-controls",
                "--config",
                args.config,
            ]
        )

    if args.command == "same-size-multiseed":
        return _run(
            [sys.executable, "-m", "src.cli", "same-size-multiseed", "--config", args.config]
        )

    if args.command == "same-size-dense-multiseed-control":
        return _run(
            [
                sys.executable,
                "-m",
                "src.cli",
                "same-size-dense-multiseed-control",
                "--config",
                args.config,
            ]
        )

    if args.command == "error-analysis":
        return _run([sys.executable, "-m", "src.cli", "error-analysis", "--config", args.config])

    if args.command == "plot-recovery":
        return _run([sys.executable, "-m", "src.cli", "plot-recovery", "--config", args.config])

    if args.command == "plot-tradeoffs":
        return _run([sys.executable, "-m", "src.cli", "plot-tradeoffs", "--config", args.config])

    if args.command == "export-final-registry":
        return _run(
            [sys.executable, "-m", "src.cli", "export-final-registry", "--config", args.config]
        )

    if args.command == "param-budget":
        return _run([sys.executable, "-m", "src.cli", "param-budget", "--config", args.config])

    if args.command == "summarize-baselines":
        return _run(
            [sys.executable, "-m", "src.cli", "summarize-baselines", "--config", args.config]
        )

    if args.command == "donor-gap-gate":
        return _run([sys.executable, "-m", "src.cli", "donor-gap-gate", "--config", args.config])

    if args.command == "freeze-mobile-actions-eval":
        return _run(
            [sys.executable, "-m", "src.cli", "freeze-mobile-actions-eval", "--config", args.config]
        )

    if args.command == "manifest-smoke":
        return _run([sys.executable, "-m", "src.cli", "manifest-smoke", "--config", args.config])

    parser.error(f"Unsupported command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
