from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.models.load_gemma import probe_gemma_loading, resolve_model_source


class LoadGemmaTests(unittest.TestCase):
    def test_resolve_model_source_requires_a_source(self) -> None:
        with self.assertRaises(ValueError):
            resolve_model_source({})

    def test_probe_reports_missing_chat_template(self) -> None:
        class DummyAutoConfig:
            @classmethod
            def from_pretrained(cls, source: str, **kwargs: object) -> object:
                return SimpleNamespace(model_type="gemma3", architectures=["Gemma3ForCausalLM"])

        class DummyAutoTokenizer:
            @classmethod
            def from_pretrained(cls, source: str, **kwargs: object) -> object:
                return SimpleNamespace(chat_template=None)

        with patch("src.models.load_gemma.Path.exists", return_value=True):
            with patch.dict(
                "sys.modules",
                {
                    "transformers": SimpleNamespace(
                        AutoConfig=DummyAutoConfig,
                        AutoTokenizer=DummyAutoTokenizer,
                    )
                },
            ):
                report = probe_gemma_loading(
                    {
                        "loader": "transformers",
                        "id": "google/gemma-3-1b-it",
                        "local_path": "E:/Model/google--gemma-3-1b-it",
                    },
                    require_chat_template=True,
                )

        self.assertEqual(report.status, "blocked")
        self.assertEqual(report.blocker_code, "chat_template_unavailable")

    def test_probe_passes_when_config_and_tokenizer_resolve(self) -> None:
        class DummyAutoConfig:
            @classmethod
            def from_pretrained(cls, source: str, **kwargs: object) -> object:
                return SimpleNamespace(model_type="gemma3", architectures=["Gemma3ForCausalLM"])

        class DummyAutoTokenizer:
            @classmethod
            def from_pretrained(cls, source: str, **kwargs: object) -> object:
                return SimpleNamespace(
                    chat_template="{{ bos_token }}",
                    apply_chat_template=lambda *a, **k: "ok",
                )

        with patch("src.models.load_gemma.Path.exists", return_value=True):
            with patch.dict(
                "sys.modules",
                {
                    "transformers": SimpleNamespace(
                        AutoConfig=DummyAutoConfig,
                        AutoTokenizer=DummyAutoTokenizer,
                    )
                },
            ):
                report = probe_gemma_loading(
                    {
                        "loader": "transformers",
                        "id": "google/gemma-3-1b-it",
                        "local_path": "E:/Model/google--gemma-3-1b-it",
                    },
                    require_chat_template=True,
                )

        self.assertEqual(report.status, "passed")
        self.assertTrue(report.metadata["chat_template_available"])


if __name__ == "__main__":
    unittest.main()
