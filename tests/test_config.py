from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.utils.config import config_hash, ensure_execution_variant, load_config


class ConfigLoaderTests(unittest.TestCase):
    def test_load_config_merges_extends_recursively(self) -> None:
        tmp_root = Path("tests/_tmp/config_loader")
        tmp_root.mkdir(parents=True, exist_ok=True)
        base_path = tmp_root / "base.json"
        child_path = tmp_root / "child.json"

        base_path.write_text(
            json.dumps(
                {
                    "execution_variant": "V24",
                    "model": {"id": "base-model", "local_files_only": True},
                    "seed": 17,
                }
            ),
            encoding="utf-8",
        )
        child_path.write_text(
            json.dumps(
                {
                    "extends": "base.json",
                    "model": {"id": "child-model"},
                    "experiment_name": "smoke",
                }
            ),
            encoding="utf-8",
        )

        loaded = load_config(child_path)

        self.assertEqual(loaded["model"]["id"], "child-model")
        self.assertTrue(loaded["model"]["local_files_only"])
        self.assertEqual(loaded["seed"], 17)
        self.assertEqual(loaded["experiment_name"], "smoke")

    def test_config_hash_is_stable(self) -> None:
        config_a = {"a": 1, "b": {"c": 2}}
        config_b = {"b": {"c": 2}, "a": 1}
        self.assertEqual(config_hash(config_a), config_hash(config_b))

    def test_ensure_execution_variant_rejects_unknown_values(self) -> None:
        config = {"execution_variant": "V99"}
        with self.assertRaises(ValueError):
            ensure_execution_variant(config)


if __name__ == "__main__":
    unittest.main()
