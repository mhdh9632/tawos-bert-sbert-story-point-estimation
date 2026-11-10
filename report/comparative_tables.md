# Comparative Results

## Five-fold temporal validation summary

| Representation | MAE mean | MAE SD | MdAE mean | MdAE SD |
|---|---:|---:|---:|---:|
| BERT | 4.5545 | 0.2809 | 2.0266 | 0.0825 |
| SBERT | 4.5884 | 0.3763 | **2.0242** | 0.2154 |
| Combined BERT + SBERT | **4.5239** | 0.2999 | 2.0961 | 0.1090 |

## Final held-out test results

| Representation | MAE | MdAE | Selected epochs | Input dimensions |
|---|---:|---:|---:|---:|
| BERT | 4.3400 | 2.1191 | 3 | 768 |
| SBERT | 4.4238 | 2.0220 | 1 | 384 |
| Combined BERT + SBERT | **3.9416** | **1.6988** | 3 | 1,152 |

## Fold-level validation results

| Representation | Fold | MAE | MdAE | Best epoch |
|---|---:|---:|---:|---:|
| BERT | 1 | 4.6205 | 1.9664 | 3 |
| BERT | 2 | 4.7768 | 1.9281 | 4 |
| BERT | 3 | 4.7716 | 2.0285 | 3 |
| BERT | 4 | 4.5105 | 2.0769 | 3 |
| BERT | 5 | 4.0931 | 2.1333 | 8 |
| SBERT | 1 | 4.8302 | 2.3262 | 1 |
| SBERT | 2 | 4.8212 | 1.7208 | 1 |
| SBERT | 3 | 4.8659 | 2.0100 | 4 |
| SBERT | 4 | 4.4291 | 1.9999 | 1 |
| SBERT | 5 | 3.9957 | 2.0642 | 6 |
| Combined | 1 | 4.7524 | 2.0580 | 3 |
| Combined | 2 | 4.7501 | 2.0236 | 4 |
| Combined | 3 | 4.6822 | 2.1779 | 4 |
| Combined | 4 | 4.3648 | 1.9808 | 1 |
| Combined | 5 | 4.0699 | 2.2403 | 2 |

