# NLP Project — Problem Statement 10

## Combined BERT and SBERT Embeddings for Temporal Story-Point Estimation

This project investigates whether combining contextual BERT representations with semantic SBERT representations improves the prediction of story points for later issues from the same software project.

**Executed Google Colab:** [Open the completed experiment](https://colab.research.google.com/drive/1IITpLs0qPoyH-MEnmVGMHTFEbK4Cu0yx?usp=sharing)

**GitHub repository:** [mhdh9632/tawos-bert-sbert-story-point-estimation](https://github.com/mhdh9632/tawos-bert-sbert-story-point-estimation)


## Main finding

The experiment automatically selected **Lsstcorp Data Management (`DM`)**, the TAWOS project with the largest eligible sample. It used 11,825 older issues for development and the newest 2,957 issues for final testing.

| Representation | Final MAE | Final MdAE |
|---|---:|---:|
| BERT | 4.3400 | 2.1191 |
| SBERT | 4.4238 | 2.0220 |
| **Combined BERT + SBERT** | **3.9416** | **1.6988** |

The combined model achieved the lowest held-out MAE and MdAE. Relative to BERT, fusion reduced MAE by **9.18%** and MdAE by **19.83%**. Relative to SBERT, it reduced MAE by **10.90%** and MdAE by **15.98%**.

## Repository contents

- `colab/TAWOS_Problem_10_Colab.ipynb` — clean reproducible notebook.
- `colab/TAWOS_Problem_10_Colab_executed.ipynb` — notebook with recorded outputs.
- `colab/TAWOS_Problem_10_Colab.py` — editable percent-format notebook source.
- `config/experiment.json` — exact model and training configuration.
- `results/` — fold metrics, summaries, plots, split metadata, and final metrics.
- `report/final_report.pdf` — submission-ready project report.
- `report/final_report.md` — editable report source.
- `DEMO.md` — demonstration and reproduction script.

## Method summary

1. Stream the 4.31 GB TAWOS SQL dump without importing MySQL.
2. Retain issue title, cleaned description, creation date, story point, and project identity.
3. Exclude missing values and issues whose title, description, or story point changed after estimation.
4. Sort the selected project chronologically.
5. Reserve the newest 20% as a final test set.
6. Use five expanding-window temporal validation folds within the older 80%.
7. Extract frozen `bert-base-uncased` and `all-MiniLM-L6-v2` embeddings.
8. Train identical three-layer regression heads for BERT, SBERT, and concatenated representations.
9. Report MAE and MdAE separately for validation and final testing.

## Reproduce in Colab

See [COLAB_README.md](COLAB_README.md) or [DEMO.md](DEMO.md). The original 4.31 GB dataset and generated embedding/model caches are intentionally excluded from Git because of their size. The notebook recreates them in Google Drive.

## Dataset

TAWOS V1.1: [official dataset DOI](https://doi.org/10.5522/04/21308124) and [schema/documentation](https://github.com/SOLAR-group/TAWOS).

The TAWOS data remains subject to its research-use terms and Apache License 2.0. Do not attempt to re-identify contributors.

