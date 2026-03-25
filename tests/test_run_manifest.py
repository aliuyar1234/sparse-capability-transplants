from __future__ import annotations

import json
import shutil
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.utils.run_manifest import (
    build_run_id,
    create_run_manifest,
    update_run_manifest,
    write_run_manifest,
)


class RunManifestTests(unittest.TestCase):
    def test_build_run_id_uses_expected_naming_convention(self) -> None:
        run_id = build_run_id(
            execution_variant="V24",
            slot_id="bootstrap",
            milestone="M0",
            experiment_name="bootstrap_smoke",
            seed=17,
            timestamp=datetime(2026, 3, 23, tzinfo=timezone.utc),
        )
        self.assertEqual(run_id, "20260323_V24_bootstrap_m0_bootstrap_smoke_s17")

    def test_manifest_write_and_update_roundtrip(self) -> None:
        config = {
            "execution_variant": "V24",
            "milestone": "M0",
            "slot_id": "bootstrap",
            "experiment_name": "bootstrap_smoke",
            "seed": 17,
            "run": {"output_root": "."},
        }
        manifest = create_run_manifest(
            config=config,
            config_path="configs/m0_smoke.json",
            command=["python"],
        )

        tmp_root = Path("tests/_tmp/run_manifest")
        tmp_root.mkdir(parents=True, exist_ok=True)
        manifest["artifact_paths"]["run_dir"] = str(tmp_root.resolve())
        run_dir = write_run_manifest(manifest)
        manifest_path = run_dir / "run_manifest.json"
        updated = update_run_manifest(manifest_path, {"status": "passed"})

        roundtrip = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(updated["status"], "passed")
        self.assertEqual(roundtrip["status"], "passed")

    def test_manifest_reuses_latest_resumable_run_with_matching_config_hash(self) -> None:
        tmp_root = Path("tests/_tmp/run_manifest_resume")
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True, exist_ok=True)
        config = {
            "execution_variant": "V24",
            "milestone": "M3",
            "slot_id": "bootstrap",
            "experiment_name": "resume_target",
            "seed": 17,
            "run": {"output_root": str(tmp_root)},
        }
        first = create_run_manifest(
            config=config,
            config_path="configs/m3_resume.json",
            command=["python"],
        )
        run_dir = write_run_manifest(first)
        update_run_manifest(run_dir / "run_manifest.json", {"status": "interrupted"})

        resumed = create_run_manifest(
            config=config,
            config_path="configs/m3_resume.json",
            command=["python"],
        )

        self.assertEqual(resumed["run_id"], first["run_id"])
        self.assertEqual(resumed["status"], "running")
        self.assertEqual(resumed["resume_count"], 1)


if __name__ == "__main__":
    unittest.main()
