from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_donor_gap_gate(config: dict[str, Any]) -> dict[str, Any]:
    analysis_config = dict(config.get("analysis", {}))
    summary_path = analysis_config.get("summary_path")
    if not summary_path:
        raise ValueError("Config is missing analysis.summary_path.")
    summary_payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    gate_payload = {
        "summary_path": str(Path(summary_path).resolve()),
        "primary_metric": summary_payload["primary_metric"],
        "gate_decision": summary_payload["gate_decision"],
    }
    return gate_payload


def write_donor_gap_gate(*, config: dict[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payload = build_donor_gap_gate(config)
    output_path = destination / "donor_gap_gate.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path
