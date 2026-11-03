"""Execute the code cells of a Jupyter notebook without Jupyter.

Cluster compute nodes do not always provide ``nbconvert``. This runner parses
the notebook JSON, executes every code cell in order in a single namespace, and
never writes back to the ``.ipynb`` (so the notebook's stored outputs and git
state are untouched). Line magics and shell escapes are skipped.

Usage::

    python scripts/run_notebook.py notebooks/8-analysis-across-datasets.ipynb

The working directory is switched to the notebook's folder, so relative paths
inside the notebook behave exactly as they do in Jupyter.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def iter_code_cells(notebook: dict):
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        yield index, "".join(cell.get("source", []))


def clean_source(source: str) -> str:
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("%", "!")):
            continue
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", help="Path to the .ipynb file to execute.")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after a failing cell instead of stopping.",
    )
    parser.add_argument(
        "--cells",
        default=None,
        help="Comma-separated cell indices to execute (default: all code cells).",
    )
    args = parser.parse_args()

    path = Path(args.notebook).resolve()
    if not path.is_file():
        sys.exit(f"notebook not found: {path}")

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.chdir(path.parent)

    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__", "__file__": str(path)}
    selected = (
        {int(part) for part in args.cells.split(",")} if args.cells else None
    )

    for index, source in iter_code_cells(notebook):
        if selected is not None and index not in selected:
            continue
        code = clean_source(source)
        if not code.strip():
            continue
        label = next(
            (line.strip() for line in code.splitlines() if line.strip()), ""
        )
        print(f"[cell {index}] {label[:100]}", flush=True)
        try:
            exec(compile(code, f"{path.name}:cell{index}", "exec"), namespace)
        except Exception:
            traceback.print_exc()
            if not args.keep_going:
                sys.exit(1)
        finally:
            # Mimic the inline backend: every cell starts with a clean figure.
            pyplot = sys.modules.get("matplotlib.pyplot")
            if pyplot is not None:
                pyplot.close("all")

    print(f"Finished {path.name}", flush=True)


if __name__ == "__main__":
    main()
