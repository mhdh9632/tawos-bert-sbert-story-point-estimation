# Google Colab workflow for Problem Statement 10

The complete experiment is in `colab/TAWOS_Problem_10_Colab.ipynb`. No local Python environment is required.

## One-time setup

1. Create this folder in Google Drive: `MyDrive/TAWOS_NLP/data/raw/`.
2. Upload the extracted `TAWOS.sql` file to that folder.
3. Upload `colab/TAWOS_Problem_10_Colab.ipynb` to Drive, or open it from GitHub after publishing this repository.
4. In Colab, select **Runtime → Change runtime type → GPU**.
5. Run the notebook from top to bottom and approve the Google Drive mount.

The first run streams the SQL dump and saves a compact CSV. Later runs reuse that CSV and the cached embeddings, so the 4.31 GB dump is not parsed again unless the cache is removed.

## Drive outputs

- `data/processed/eligible_issues.csv`: leakage-safe labelled issues.
- `data/processed/project_profile.csv`: eligible count and label summary per project.
- `data/processed/<PROJECT>/development.csv`: chronological older 80%.
- `data/processed/<PROJECT>/test.csv`: chronological newest 20%.
- `data/processed/<PROJECT>/split_manifest.json`: exact temporal fold boundaries.
- `artifacts/embeddings/<PROJECT>/`: cached BERT and SBERT arrays.
- `artifacts/models/<PROJECT>/`: trained regression heads and scalers.
- `results/`: validation/test tables, predictions, plots, and experiment metadata.

The notebook defaults to the project with the largest eligible sample. To use a specific assigned project, set `PROJECT_KEY` near the top, for example `PROJECT_KEY = "MESOS"`.

