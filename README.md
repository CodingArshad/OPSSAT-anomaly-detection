# Satellite Telemetry Anomaly Detection (OPSSAT-AD)

A binary classifier that labels real satellite telemetry fragments as anomalous or nominal, built on the OPSSAT-AD dataset — actual on-orbit telemetry from ESA's OPS-SAT CubeSat. The dataset contains 2,123 labeled single-channel telemetry fragments across 9 channels, roughly 20% of them anomalous. The end goal is a complete, honest ML workflow on real flight data (load → clean → explore → split → train → evaluate), with evaluation done correctly under class imbalance — a model that always predicts "normal" scores ~80% accuracy here while detecting nothing, so accuracy alone is never the metric.

## Current Version (V2)

V2 is the reasoning version: exploratory data analysis on the raw fragments before any features or models exist. The question this version answers is "what does an anomaly actually look like, per channel?" because the channels are different telemetry streams I don't expect anomalies to look the same across them.

### Features
- Verified data loading with fact assertions (2,123 fragments / 9 channels / ~20% anomalous) — V1
- Cleaning with logged decisions on NaNs, duplicates, non-numeric values, empty fragments — V1
- `run_eda()` pass 1, producing saved figures:
  - Per-channel fragment counts and anomaly rates (this figure shows where the anomalies live)
  - Side-by-side anomalous-vs-nominal fragment line plots for each of the 9 channels, so an anomalous fragment can be visually compared against a nominal one(from the same channel)
- All figures saved to `figures/` with descriptive filenames

## Technologies
- Python 3
- pandas, NumPy
- Matplotlib (EDA figures)

## Project Structure
```
OPSSAT-AD/
├── data/
│   └── raw/            # dataset.csv, segments.csv (Zenodo record 12588359, not committed)
├── figures/            # saved EDA output
├── src/
│   ├── load.py         # load_fragments
│   ├── clean.py        # clean_fragments
│   └── eda.py          # run_eda (pass 1: raw fragments)
├── main.py             # load → clean → EDA pass 1
├── docs/
│   └── algorithm.md
└── README.md
```

## Long-Term Goals
- V3: Feature extraction (summary statistics per fragment), EDA pass 2 on those features, and a stratified train/test split
- V4: First evaluated model — a constrained Decision Tree, scored on precision/recall/F1
- V5: Logistic Regression added, both models compared, error modes interpreted
- V6: Final documentation pass and public GitHub release
