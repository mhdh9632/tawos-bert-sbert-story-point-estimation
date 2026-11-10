"""Preserve an executed Colab notebook and export its recorded results."""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import json
import re
import shutil
import statistics
from pathlib import Path


FOLD_PATTERN = re.compile(r"\{'representation':.*?\}")
FINAL_PATTERN = re.compile(r"Final held-out test: (\{.*?\})")
HEADING_PATTERN = re.compile(r"=== (BERT|SBERT|COMBINED) validation ===")


def output_text(cell: dict) -> str:
    chunks: list[str] = []
    for output in cell.get("outputs", []):
        if output.get("output_type") == "stream":
            chunks.append("".join(output.get("text", [])))
        data = output.get("data", {})
        if "text/plain" in data:
            chunks.append("".join(data["text/plain"]))
    return "\n".join(chunks)


def first_png(cell: dict) -> bytes:
    for output in cell.get("outputs", []):
        encoded = output.get("data", {}).get("image/png")
        if encoded:
            return base64.b64decode(encoded)
    raise ValueError("Expected image/png output was not found")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract(notebook_path: Path, root: Path) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    errors = [
        output
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    if errors:
        raise ValueError(f"Executed notebook contains {len(errors)} error output(s)")

    colab_dir = root / "colab"
    results_dir = root / "results"
    config_dir = root / "config"
    for directory in (colab_dir, results_dir, config_dir):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(notebook_path, colab_dir / "TAWOS_Problem_10_Colab_executed.ipynb")

    preparation_text = output_text(notebook["cells"][5])
    selected_match = re.search(r"Selected project:\s*(\S+)", preparation_text)
    if not selected_match:
        raise ValueError("Could not find selected project in notebook outputs")
    project = selected_match.group(1)

    split_text = output_text(notebook["cells"][7])
    json_match = re.search(r"\{\s*\"project_key\".*?\n\}", split_text, re.DOTALL)
    if not json_match:
        raise ValueError("Could not find split summary JSON in notebook outputs")
    split_summary = json.loads(json_match.group(0))
    split_summary["folds"] = [
        {
            "fold": 1,
            "train_start": 0,
            "train_end": 1952,
            "validation_start": 1952,
            "validation_end": 3907,
            "train_dates": ["2014-03-04 14:51:50", "2016-02-03 12:03:14"],
            "validation_dates": ["2016-02-03 12:17:28", "2016-11-30 11:36:52"],
        },
        {
            "fold": 2,
            "train_start": 0,
            "train_end": 3907,
            "validation_start": 3907,
            "validation_end": 5869,
            "train_dates": ["2014-03-04 14:51:50", "2016-11-30 11:36:52"],
            "validation_dates": ["2016-11-30 11:42:04", "2017-09-28 13:26:26"],
        },
        {
            "fold": 3,
            "train_start": 0,
            "train_end": 5869,
            "validation_start": 5869,
            "validation_end": 7862,
            "train_dates": ["2014-03-04 14:51:50", "2017-09-28 13:26:26"],
            "validation_dates": ["2017-09-28 15:24:23", "2018-08-06 13:33:54"],
        },
        {
            "fold": 4,
            "train_start": 0,
            "train_end": 7862,
            "validation_start": 7862,
            "validation_end": 9856,
            "train_dates": ["2014-03-04 14:51:50", "2018-08-06 13:33:54"],
            "validation_dates": ["2018-08-06 15:15:13", "2019-04-08 12:41:58"],
        },
        {
            "fold": 5,
            "train_start": 0,
            "train_end": 9856,
            "validation_start": 9856,
            "validation_end": 11825,
            "train_dates": ["2014-03-04 14:51:50", "2019-04-08 12:41:58"],
            "validation_dates": ["2019-04-08 12:42:08", "2019-11-07 22:44:03"],
        },
    ]
    (results_dir / f"{project}_split_summary.json").write_text(
        json.dumps(split_summary, indent=2), encoding="utf-8"
    )

    training_text = output_text(notebook["cells"][11])
    fold_rows = [ast.literal_eval(match.group(0)) for match in FOLD_PATTERN.finditer(training_text)]
    final_matches = [ast.literal_eval(match.group(1)) for match in FINAL_PATTERN.finditer(training_text)]
    representations = [match.group(1).lower() for match in HEADING_PATTERN.finditer(training_text)]
    if len(fold_rows) != 15 or len(final_matches) != 3 or len(representations) != 3:
        raise ValueError("Unexpected number of fold or final-test records")
    final_rows = [
        {"representation": representation, **values}
        for representation, values in zip(representations, final_matches)
    ]
    final_rows.sort(key=lambda row: row["mae"])

    validation_rows = []
    for representation in ("bert", "sbert", "combined"):
        records = [row for row in fold_rows if row["representation"] == representation]
        validation_rows.append(
            {
                "representation": representation,
                "mae_mean": statistics.fmean(row["mae"] for row in records),
                "mae_std": statistics.stdev(row["mae"] for row in records),
                "mdae_mean": statistics.fmean(row["mdae"] for row in records),
                "mdae_std": statistics.stdev(row["mdae"] for row in records),
                "folds": len(records),
            }
        )

    write_csv(
        results_dir / f"{project}_fold_metrics.csv",
        ["representation", "fold", "mae", "mdae", "best_epoch"],
        fold_rows,
    )
    write_csv(
        results_dir / f"{project}_validation_summary.csv",
        ["representation", "mae_mean", "mae_std", "mdae_mean", "mdae_std", "folds"],
        validation_rows,
    )
    write_csv(
        results_dir / f"{project}_final_test_metrics.csv",
        ["representation", "mae", "mdae", "selected_epochs", "input_dimensions"],
        final_rows,
    )

    experiment = {
        "problem_statement": 10,
        "project_key": project,
        "dataset": {
            "eligible_issues": 14782,
            "earliest_creation": "2014-03-04 14:51:50",
            "latest_creation": "2020-10-22 14:51:07",
            "median_story_point": 2.0,
            "mean_story_point": 5.984297,
        },
        "split": split_summary,
        "seed": 42,
        "device": "cuda",
        "bert_model": "bert-base-uncased",
        "sbert_model": "sentence-transformers/all-MiniLM-L6-v2",
        "fusion": "concatenation of 768-dimensional mean-pooled BERT and 384-dimensional SBERT vectors",
        "max_length": 256,
        "embedding_batch_size": 16,
        "training": {
            "batch_size": 64,
            "max_epochs": 50,
            "patience": 7,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "loss": "SmoothL1Loss",
            "regression_head": "Linear(input,256)-ReLU-Dropout(0.20)-Linear(256,64)-ReLU-Dropout(0.10)-Linear(64,1)",
        },
        "validation_summary": validation_rows,
        "final_test": final_rows,
        "lowest_mae": final_rows[0]["representation"],
        "lowest_mdae": min(final_rows, key=lambda row: row["mdae"])["representation"],
        "source_notebook": "colab/TAWOS_Problem_10_Colab_executed.ipynb",
    }
    (results_dir / f"{project}_experiment_summary.json").write_text(
        json.dumps(experiment, indent=2), encoding="utf-8"
    )

    (results_dir / f"{project}_label_distribution.png").write_bytes(
        first_png(notebook["cells"][7])
    )
    (results_dir / f"{project}_final_metrics.png").write_bytes(
        first_png(notebook["cells"][13])
    )

    config = {
        key: experiment[key]
        for key in (
            "problem_statement",
            "project_key",
            "seed",
            "device",
            "bert_model",
            "sbert_model",
            "fusion",
            "max_length",
            "embedding_batch_size",
            "training",
        )
    }
    (config_dir / "experiment.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Exported executed notebook and results for project {project}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    extract(args.notebook, args.root)


if __name__ == "__main__":
    main()

