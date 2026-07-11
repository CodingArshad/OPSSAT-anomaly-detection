# Satellite Telemetry Anomaly Detection (OPSSAT-AD)

A binary classifier that labels real satellite telemetry fragments as anomalous or nominal, built on the OPSSAT-AD dataset (actual on-orbit telemetry from ESA's OPS-SAT CubeSat). The dataset contains 2,123 labeled single-channel telemetry fragments across 9 channels, roughly 20% of them anomalous. The end goal is a complete, honest ML workflow on real flight data (load → clean → explore → split → train → evaluate), with evaluation done correctly under class imbalance (a model that always predicts "normal" scores ~80% accuracy here while detecting nothing, so accuracy alone is never the metric).

## Current Version (V3)

V3 turns variable-length telemetry fragments into a fixed-width feature matrix that models can consume, checks whether those features actually separate the classes (EDA pass 2), and locks in a stratified train/test split. Everything is now in place for modeling but no model exists yet.

### Features
- Verified data loading with published-fact assertions (2,123 fragments / 9 channels / ~20% anomalous) — V1
- Cleaning audit with logged decisions — V1
- Raw-fragment EDA: per-channel counts/anomaly rates, anomalous-vs-nominal fragment plots — V2
- `extract_features()`: ~12–15 NumPy summary statistics per fragment (mean/std/min/max/range, sample count, uniqueness/duplicate ratio, first-difference stats), plus encoded channel — producing `features_df`
- `run_eda()` pass 2: feature distributions split by label, showing which features visibly separate anomalous from nominal fragments
- `split_data()`: stratified ~80/20 train/test split with fixed `random_state`; class balance of both splits verified and printed

## Technologies
- Python 3
- pandas, NumPy
- Matplotlib (EDA figures)
- scikit-learn (`train_test_split`)

## Project Structure
```
OPSSAT-AD/
├── data/
│   └── raw/            # dataset.csv, segments.csv (Zenodo record 12588359, not committed)
├── figures/            # saved EDA output (pass 1 + pass 2)
├── src/
│   ├── load.py         # load_fragments
│   ├── clean.py        # clean_fragments
│   ├── eda.py          # run_eda (pass 1: fragments; pass 2: features)
│   ├── features.py     # extract_features
│   └── split.py        # split_data
├── main.py             # load → clean → EDA-1 → features → EDA-2 → split
├── docs/
│   └── algorithm.md
└── README.md
```

## Long-Term Goals
- V4: First evaluated model (constrained Decision Tree, scored on precision/recall/F1 against the ~80% always-normal baseline)
- V5: Logistic Regression added, both models compared, error modes interpreted
- V6: Final documentation pass and public GitHub release
