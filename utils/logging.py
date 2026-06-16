"""Lightweight logging utilities for training runs."""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any


def get_logger(name: str = "spiking_lm", level: int = logging.INFO) -> logging.Logger:
    """Create or fetch a console logger with a consistent format.

    Args:
        name: Logger name.
        level: Logging level.

    Returns:
        A configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


class MetricLogger:
    """Append training metrics to a CSV file (and optionally echo to console).

    A single flat schema is used: each ``log`` call records one row keyed by
    step. Unknown columns encountered after the header is written are stored in
    a JSON ``extra`` column to keep the CSV well-formed.
    """

    def __init__(self, log_dir: str | Path, filename: str = "metrics.csv") -> None:
        """Initialize the metric logger.

        Args:
            log_dir: Directory in which to write the metrics file.
            filename: Name of the CSV file.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / filename
        self._fieldnames: list[str] | None = None

    def log(self, metrics: dict[str, Any]) -> None:
        """Append a row of metrics to the CSV file.

        Args:
            metrics: Mapping of metric name to value for this step.
        """
        write_header = not self.path.exists()
        if self._fieldnames is None:
            self._fieldnames = list(metrics.keys())

        row: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for key, value in metrics.items():
            if key in self._fieldnames:
                row[key] = value
            else:
                extra[key] = value
        if extra:
            row["extra"] = json.dumps(extra)

        fieldnames = list(self._fieldnames)
        if extra and "extra" not in fieldnames:
            fieldnames.append("extra")

        with self.path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
