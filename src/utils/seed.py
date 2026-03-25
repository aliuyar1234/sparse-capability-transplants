from __future__ import annotations

import os
import random
from dataclasses import dataclass

PROJECT_SEEDS = (17, 29, 43)


@dataclass(frozen=True)
class SeedReport:
    seed: int
    numpy_seeded: bool
    torch_seeded: bool


def set_seed(seed: int, *, deterministic_torch: bool = False) -> SeedReport:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    numpy_seeded = False
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - optional dependency
        np = None
    if np is not None:
        np.random.seed(seed)
        numpy_seeded = True

    torch_seeded = False
    try:
        import torch
    except ImportError:  # pragma: no cover - optional dependency
        torch = None
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        torch_seeded = True

    return SeedReport(seed=seed, numpy_seeded=numpy_seeded, torch_seeded=torch_seeded)
