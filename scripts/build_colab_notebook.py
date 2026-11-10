"""Convert the percent-format Colab source into a self-contained .ipynb file."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "colab/TAWOS_Problem_10_Colab.py"
TARGET = ROOT / "colab/TAWOS_Problem_10_Colab.ipynb"


def build() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    parts = re.split(r"^# %%([^\n]*)\n", text, flags=re.MULTILINE)
    cells = []
    preamble = parts[0].strip()
    if preamble:
        raise ValueError("Notebook source must begin with a # %% marker")
    for index in range(1, len(parts), 2):
        marker = parts[index].strip()
        body = parts[index + 1]
        if marker == "[markdown]":
            lines = []
            for line in body.splitlines(keepends=True):
                if line.startswith("# "):
                    lines.append(line[2:])
                elif line.startswith("#"):
                    lines.append(line[1:])
                else:
                    lines.append(line)
            cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": body.splitlines(keepends=True),
                }
            )
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": TARGET.name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    TARGET.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Wrote {TARGET} with {len(cells)} cells")


if __name__ == "__main__":
    build()

