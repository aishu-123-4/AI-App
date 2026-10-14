# 04 · MalScan — Static PE Malware Classifier  🧬🛡️

An **ML × Security** project: classify Windows executables as **benign vs.
malware** using only **static PE-header features** — no file is ever executed,
so the whole pipeline is safe to run anywhere. This mirrors how real endpoint
scanners triage files before detonation in a sandbox.

## Dataset

**ClaMP** (Classification of Malware with PE headers): **5,210** samples
(2,722 malware / 2,488 benign), **69 features** extracted from PE headers —
DOS-header fields, file-header characteristics, optional-header fields, section
entropy, file size, and the **packer family** (`packer_type`, a 40-value
categorical; packers are a classic malware tell).

## Results

Gradient boosting (`HistGradientBoostingClassifier`), 75/25 stratified split,
plus 5-fold cross-validation:

| Metric | Score |
|--------|-------|
| Accuracy | **0.990** |
| Precision | 0.99 |
| Recall | 0.99 |
| F1 | 0.99 |
| ROC-AUC | 0.9997 |
| **CV F1 (5-fold)** | **0.991 ± 0.003** |

Confusion matrix `[[TN, FP], [FN, TP]]`: `[[613, 9], [4, 677]]`
→ 13 errors out of 1,303 test files.

**Top features (permutation importance):** `FH_char12` (file-header
characteristics flag), `Subsystem`, `filesize`, `CheckSum`, section entropy
(`E_data`, `E_file`), `AddressOfEntryPoint`. These are exactly the fields
malware analysts look at — e.g. an invalid/zero `CheckSum`, unusual subsystem,
or abnormal entry point.

### Is 99% too good?

No — and here's why that matters. Unlike a synthetic dataset, ClaMP is real and
malware vs. benign PE headers are **genuinely highly separable** (packed/obfuscated
binaries, abnormal section entropy, malformed headers). The **5-fold CV F1 of
0.991 ± 0.003** — a tight spread with no single lucky split — confirms the model
generalizes rather than memorizes, and the result is in line with published
ClaMP baselines. Feature importances computed via *permutation* (not just the
model's internal Gini) are an extra honesty check.

## Quickstart

```bash
cd 04-malscan-pe-classifier
pip install -r requirements.txt

python -m src.download_data   # fetch ClaMP CSV into data/
python -m src.train           # train + evaluate, write models/ + metrics.json
python -m src.predict 10      # classify 10 random samples (OK / XX)
```

## How it works

```
PE sample (69 features)
   ├── packer_type (categorical) ─► OneHotEncoder(handle_unknown=ignore)
   └── 68 numeric header fields   ─► passthrough (trees need no scaling)
                                      └─► HistGradientBoosting ─► P(malware)
```

- `src/data.py` — load ClaMP, split features/label, identify categorical vs numeric.
- `src/train.py` — pipeline, metrics, 5-fold CV, permutation importance, persistence.
- `src/predict.py` — load model and classify feature rows.

`handle_unknown="ignore"` on the packer encoder means a never-before-seen packer
family at inference time won't crash the model (tested).

## Project layout

```
04-malscan-pe-classifier/
├── README.md
├── requirements.txt
├── data/                 # downloaded ClaMP CSV (git-ignored)
├── models/               # saved model + metrics.json (git-ignored)
└── src/
    ├── download_data.py
    ├── data.py
    ├── train.py
    └── predict.py
tests/
└── test_pipeline.py      # 4 tests (synthetic frame, no download needed)
```

## Tests

```bash
python -m pytest -q      # 4 tests
```

## Dataset & credit

ClaMP by Ajit Kumar (`urwithajit9/ClaMP`), features extracted from PE headers.
Used here for research/education. **Scope:** static analysis only; this project
neither downloads nor executes real malware binaries.

## Possible extensions

- Scale up to **EMBER** (1.1M samples, 2,381 features) with LightGBM.
- Add a real PE feature extractor (`pefile`/`lief`) so you can score your own
  `.exe` files from disk.
- Calibrate probabilities and plot a precision-recall curve for a chosen
  operating point (false positives are costly in endpoint security).
- SHAP explanations per file for analyst-facing triage.
