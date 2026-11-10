# Demonstration and Reproduction Guide

## Short demonstration (5–7 minutes)

1. Open the [executed Colab notebook](https://colab.research.google.com/drive/1IITpLs0qPoyH-MEnmVGMHTFEbK4Cu0yx?usp=sharing).
2. Show the configuration cell: project selection, seed 42, model names, and temporal settings.
3. Show the project profile and explain why `DM` was selected: 14,782 eligible issues, the largest valid project sample.
4. Show the temporal split:
   - 11,825 development issues from 2014-03-04 to 2019-11-07 22:44;
   - 2,957 held-out test issues from 2019-11-07 22:50 to 2020-10-22.
5. Show the five expanding-window folds and verify each validation period occurs after its training period.
6. Show the embedding dimensions: BERT 768, SBERT 384, and combined 1,152.
7. Show the validation outputs and final comparison chart.
8. State the main result: combined embeddings achieved MAE 3.9416 and MdAE 1.6988, the lowest final-test errors.
9. Close with limitations: one project, frozen encoders, 256-token truncation, one seed, and no significance test.

## Full reproduction

1. In Google Drive, create `MyDrive/TAWOS_NLP/data/raw/`.
2. Place the extracted `TAWOS.sql` file in that folder.
3. Open `colab/TAWOS_Problem_10_Colab.ipynb` in Google Colab.
4. Choose **Runtime → Change runtime type → GPU**.
5. Run all notebook cells in order.
6. Approve Google Drive access when prompted.

The first run creates a compact eligible-issue CSV and embedding caches. Later runs reuse these files unless the configuration or selected issue texts change.

## Expected Google Drive outputs

```text
MyDrive/TAWOS_NLP/
├── data/processed/DM/
│   ├── development.csv
│   ├── test.csv
│   └── split_manifest.json
├── artifacts/
│   ├── embeddings/DM/
│   └── models/DM/
└── results/
    ├── DM_fold_metrics.csv
    ├── DM_validation_summary.csv
    ├── DM_final_test_metrics.csv
    ├── DM_test_predictions.csv
    ├── DM_experiment_summary.json
    └── DM_final_metrics.png
```

## Verification checks

- The executed notebook contains no error output.
- The latest development timestamp is earlier than the earliest test timestamp.
- Every fold’s training end is earlier than its validation start.
- All three representations use the same issue partitions.
- Validation and final-test results are reported separately.
- The final test set does not influence early stopping or epoch selection.

