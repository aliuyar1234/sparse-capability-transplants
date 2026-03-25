from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.models.smoke import probe_model_loading


class SmokeProbeTests(unittest.TestCase):
    def test_probe_blocks_cleanly_for_unsupported_loader(self) -> None:
        result = probe_model_loading({"loader": "custom"})
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.blocker_code, "unsupported_loader")

    def test_probe_requires_model_source(self) -> None:
        result = probe_model_loading({"loader": "transformers"})
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.blocker_code, "missing_model_source")

    def test_probe_blocks_cleanly_for_missing_local_path(self) -> None:
        result = probe_model_loading(
            {
                "loader": "transformers",
                "id": "google/gemma-3-1b-it",
                "local_path": "E:/Model/does-not-exist",
            }
        )
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.blocker_code, "local_path_missing")

    def test_probe_prefers_local_path_when_available(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []

        class DummyAutoConfig:
            @classmethod
            def from_pretrained(cls, source: str, **kwargs: object) -> object:
                calls.append((source, kwargs))
                return SimpleNamespace(
                    model_type="gemma3",
                    architectures=["Gemma3ForConditionalGeneration"],
                )

        with patch("src.models.smoke.Path.exists", return_value=True):
            with patch.dict(
                "sys.modules",
                {"transformers": SimpleNamespace(AutoConfig=DummyAutoConfig)},
            ):
                result = probe_model_loading(
                    {
                        "loader": "transformers",
                        "id": "google/gemma-3-1b-it",
                        "local_path": "E:/Model/google--gemma-3-1b-it",
                        "local_files_only": True,
                    }
                )

        self.assertEqual(result.status, "passed")
        self.assertTrue(result.metadata["used_local_path"])
        self.assertEqual(calls[0][0], "E:\\Model\\google--gemma-3-1b-it")


if __name__ == "__main__":
    unittest.main()
