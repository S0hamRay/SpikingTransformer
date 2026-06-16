"""Pytest configuration.

Placing this file at the repository root ensures the root directory is on
``sys.path`` so tests can import both the new packages (``model``, ``data``,
...) and the original modules (``attention``, ``spiking_neuron``).
"""

from __future__ import annotations
