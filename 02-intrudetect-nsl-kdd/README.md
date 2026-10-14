# 02 · IntruDetect — Network Intrusion Detection (NSL-KDD)  🛰️🛡️

An **ML × Security** project on the **NSL-KDD** benchmark — the standard dataset
for network intrusion detection. Given the features of a single network
connection, classify it as **normal vs. attack** (binary) and into one of the
four attack families: **DoS, Probe, R2L, U2R** (multi-class).

## Why NSL-KDD

NSL-KDD is the cleaned successor to KDD Cup '99 (redundant records removed).
Crucially, its **test set is drawn from a different attack distribution than the
train set** — on purpose. That makes it a realistic test of *generalization* to
novel attacks, and the gap between train and test accuracy is the whole point of
the benchmark. Reporting that gap honestly (rather than a flattering CV number)
is what shows you understand the dataset.

## Results (official train/test split)

Trained on **125,973** connections, evaluated on the held-out **22,544**-row
official test set (RandomForest + one-hot categoricals + scaled numerics):

| Task | Train acc | **Test acc** | Macro-F1 |
|------|-----------|--------------|----------|
| Binary (normal vs attack) | 0.999 | **0.777** | 0.776 |
| Multi-class (5 classes)   | 0.999 | **0.755** | 0.562 |

Per-class test performance (multi-class):

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| normal | 0.66 | 0.97 | 0.78 | 9,711 |
| DoS    | 0.96 | 0.76 | 0.85 | 7,460 |
| Probe  | 0.84 | 0.69 | 0.76 | 2,421 |
| R2L    | 0.96 | 0.07 | 0.14 | 2,885 |
| U2R    | 0.50 | 0.19 | 0.28 | 67 |

### Reading these numbers

- **The ~22-point train→test drop is expected** — it's the NSL-KDD
  generalization gap, not overfitting I failed to notice. A near-100% test score
  here would mean a data-leak bug.
- **R2L and U2R are hard** (recall 0.07 / 0.19). These rare attack types barely
  appear in training and look like normal traffic — this is the central, well-
  documented NSL-KDD challenge. High precision but low recall tells you the model
  is conservative: when it *does* flag R2L it's almost always right, but it misses
  most of them.
- **DoS and Probe are strong** (F1 0.85 / 0.76) because they have distinctive,
  well-represented signatures.

These results are consistent with published NSL-KDD baselines.

## Quickstart

```bash
cd 02-intrudetect-nsl-kdd
pip install -r requirements.txt

python -m src.download_data   # fetch KDDTrain+/KDDTest+ into data/
python -m src.train           # train both models, write models/ + metrics.json
python -m src.predict 10      # classify the first 10 test rows (OK / XX)
```

## How it works

```
connection record (41 features)
   ├── 3 categorical (protocol_type, service, flag) ─► OneHotEncoder(handle_unknown=ignore)
   └── 38 numeric                                    ─► StandardScaler
                                                         └─► RandomForest ─► class
```

- `src/preprocess.py` — column schema, the standard 40→4 attack-category map,
  and both target columns (`is_attack`, `category`).
- `src/train.py` — shared `ColumnTransformer` pipeline, trains binary +
  multi-class, writes per-class metrics and confusion matrices.
- `src/predict.py` — loads a saved model and classifies records.

The `handle_unknown="ignore"` encoder matters: the test set contains service
values never seen in training, which would otherwise crash inference.

## Project layout

```
02-intrudetect-nsl-kdd/
├── README.md
├── requirements.txt
├── data/                     # downloaded NSL-KDD files (git-ignored)
├── models/                   # saved models + metrics.json (git-ignored)
└── src/
    ├── download_data.py
    ├── preprocess.py
    ├── train.py
    └── predict.py
tests/
└── test_preprocess.py        # 7 tests (no download required)
```

## Tests

```bash
python -m pytest -q      # 7 tests
```

## Dataset & credit

NSL-KDD, University of New Brunswick (CIC). Fetched from a public mirror of the
canonical `KDDTrain+` / `KDDTest+` files. Used here for research/education.

## Possible extensions

- Address R2L/U2R recall with SMOTE / class re-weighting / anomaly-style models.
- Compare RandomForest with gradient boosting and a small MLP.
- Add a one-vs-rest ROC-AUC per attack class and a feature-importance report.
