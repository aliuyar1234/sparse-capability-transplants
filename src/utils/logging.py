from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure_logging(level: str = "INFO", *, log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger("sparse_capability_transplants")
    logger.setLevel(level.upper())
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
