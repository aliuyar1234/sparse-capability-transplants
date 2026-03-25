from __future__ import annotations

import random
import unittest

from src.utils.seed import PROJECT_SEEDS, set_seed


class SeedTests(unittest.TestCase):
    def test_project_seed_set_is_locked(self) -> None:
        self.assertEqual(PROJECT_SEEDS, (17, 29, 43))

    def test_set_seed_replays_python_random(self) -> None:
        set_seed(17)
        first = [random.random() for _ in range(3)]
        set_seed(17)
        second = [random.random() for _ in range(3)]
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
