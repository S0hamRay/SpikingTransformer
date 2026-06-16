"""Configuration loading and device selection helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file into a dictionary.

    Args:
        path: Path to the YAML config.

    Returns:
        The parsed configuration as a nested dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping, got {type(config)!r}.")
    return config


def select_device(preference: str = "auto") -> torch.device:
    """Select the best available compute device.

    Args:
        preference: One of ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.

    Returns:
        The resolved :class:`torch.device`.
    """
    preference = preference.lower()
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
