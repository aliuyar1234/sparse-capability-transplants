from __future__ import annotations

import json
import unittest
from dataclasses import dataclass
from pathlib import Path

from src.eval.golden_fixtures import write_golden_fixture


@dataclass(frozen=True)
class DummyPayload:
    value: int


class GoldenFixturesTests(unittest.TestCase):
    def test_write_golden_fixture_serializes_dataclasses(self) -> None:
        output_path = Path("tests/_tmp/golden/fixture.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        written = write_golden_fixture(
            fixture_payload={"payload": DummyPayload(value=3)},
            output_path=output_path,
        )
        payload = json.loads(written.read_text(encoding="utf-8"))
        self.assertEqual(payload["payload"]["value"], 3)


if __name__ == "__main__":
    unittest.main()
