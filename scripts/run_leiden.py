#!/usr/bin/env python3
"""Run Leiden community detection on the Neo4j Concept co-occurrence graph."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/run_leiden.py` from the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from db.config import leiden_gamma
from db.leiden import LeidenError, run_leiden


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description=(
            "Cluster Concepts with Neo4j GDS Leiden on CO_OCCURS_WITH, "
            "materialize Community nodes, and cascade Papers."
        )
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help=(
            "Leiden resolution parameter (higher → more communities). "
            f"Default: LEIDEN_GAMMA env or {leiden_gamma()}."
        ),
    )
    args = parser.parse_args()

    try:
        run_leiden(gamma=args.gamma)
    except LeidenError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
