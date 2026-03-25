from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class HeartbeatPaths:
    heartbeat_path: Path
    progress_path: Path


class RunHeartbeat:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        phase: str,
        total_units: int | None,
        unit_name: str,
        heartbeat_interval_seconds: float = 10.0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.phase = phase
        self.total_units = total_units
        self.unit_name = unit_name
        self.heartbeat_interval_seconds = max(0.0, heartbeat_interval_seconds)
        self.started_at_monotonic = time.monotonic()
        self.last_write_monotonic = 0.0
        self.last_completed_units = -1
        self.paths = HeartbeatPaths(
            heartbeat_path=self.output_dir / "heartbeat.json",
            progress_path=self.output_dir / "progress.json",
        )

    def _write(
        self,
        *,
        status: str,
        completed_units: int,
        message: str | None = None,
        metrics: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
        force: bool,
    ) -> dict[str, Any]:
        now_monotonic = time.monotonic()
        if not force and self.heartbeat_interval_seconds > 0.0:
            if (now_monotonic - self.last_write_monotonic) < self.heartbeat_interval_seconds:
                return {}

        elapsed_seconds = max(0.0, now_monotonic - self.started_at_monotonic)
        throughput = (
            completed_units / elapsed_seconds
            if elapsed_seconds > 0.0 and completed_units > 0
            else None
        )
        eta_seconds = None
        percent_complete = None
        if self.total_units is not None and self.total_units > 0:
            remaining = max(self.total_units - completed_units, 0)
            percent_complete = completed_units / self.total_units
            if throughput and throughput > 0.0:
                eta_seconds = remaining / throughput

        payload = {
            "status": status,
            "phase": self.phase,
            "updated_at_utc": utc_now_iso(),
            "elapsed_seconds": elapsed_seconds,
            "unit_name": self.unit_name,
            "completed_units": completed_units,
            "total_units": self.total_units,
            "percent_complete": percent_complete,
            "throughput_units_per_second": throughput,
            "eta_seconds": eta_seconds,
            "message": message,
            "metrics": metrics or {},
            "extra": extra or {},
        }
        serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        self.paths.heartbeat_path.write_text(serialized, encoding="utf-8")
        self.paths.progress_path.write_text(serialized, encoding="utf-8")
        self.last_write_monotonic = now_monotonic
        self.last_completed_units = completed_units
        return payload

    def start(
        self,
        *,
        completed_units: int = 0,
        message: str | None = None,
        metrics: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._write(
            status="running",
            completed_units=completed_units,
            message=message,
            metrics=metrics,
            extra=extra,
            force=True,
        )

    def maybe_update(
        self,
        *,
        completed_units: int,
        message: str | None = None,
        metrics: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        return self._write(
            status="running",
            completed_units=completed_units,
            message=message,
            metrics=metrics,
            extra=extra,
            force=force,
        )

    def mark_interrupted(
        self,
        *,
        completed_units: int,
        message: str | None = None,
        metrics: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._write(
            status="interrupted",
            completed_units=completed_units,
            message=message,
            metrics=metrics,
            extra=extra,
            force=True,
        )

    def mark_completed(
        self,
        *,
        completed_units: int,
        message: str | None = None,
        metrics: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._write(
            status="passed",
            completed_units=completed_units,
            message=message,
            metrics=metrics,
            extra=extra,
            force=True,
        )

    def mark_failed(
        self,
        *,
        completed_units: int,
        message: str | None = None,
        metrics: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._write(
            status="failed",
            completed_units=completed_units,
            message=message,
            metrics=metrics,
            extra=extra,
            force=True,
        )
