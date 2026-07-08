# Satellite Telemetry Anomaly Detection (OPSSAT-AD)

A binary classifier that labels real satellite telemetry fragments as anomalous or nominal, built on the OPSSAT-AD dataset — actual on-orbit telemetry from ESA's OPS-SAT CubeSat. The dataset contains 2,123 labeled single-channel telemetry fragments across 9 channels, roughly 20% of them anomalous. The end goal is a complete, honest ML workflow on real flight data (load → clean → explore → split → train → evaluate), with evaluation done correctly under class imbalance — a model that always predicts "normal" scores ~80% accuracy here while detecting nothing, so accuracy alone is never the metric.

## Current Version (V1 — in progress)

V1 is the data foundation: get the raw Zenodo archive loaded into a structured DataFrame and verified clean. No plots, no features, no models yet — the "it works" signal for this version is that the loader's sanity checks pass against the dataset's published facts (2,123 fragments, 9 channels, ~20% anomaly rate).

The raw archive (`data/raw/dataset.csv`, `data/raw/segments.csv`) is downloaded and confirmed to match its own on-disk layout (documented in `docs/algorithm.md`). Nothing is implemented yet — `src/load.py`, `src/clean.py`, and `main.py` are stubs to be built out next.

### Planned Features (V1)
- `load_fragments()` merges `dataset.csv` and `segments.csv` into a single `fragments_df` — one row per telemetry fragment, with fragment ID, channel, raw value array, timestamps, and anomaly label
- Load-time assertions verify 2,123 fragments, 9 channels, and a ~20% positive (anomalous) rate before anything downstream runs
- `clean_fragments()` audits the loaded data: duplicate fragment IDs, NaN/non-numeric values inside fragments, empty fragments — and logs every decision made

## Technologies
- Python 3
- pandas (fragment table)
- NumPy (per-fragment value arrays)

## Project Structure
```
OPSSAT-AD/
├── data/
│   └── raw/            # dataset.csv, segments.csv (Zenodo record 12588359, not committed)
├── src/
│   ├── load.py         # load_fragments
│   └── clean.py        # clean_fragments
├── main.py             # runs load → clean, prints sanity-check summary
├── docs/
│   └── algorithm.md
└── README.md
```

## Long-Term Goals
- V2: Exploratory data analysis on raw fragments — per-channel counts, anomaly rates, and what an anomalous fragment actually looks like next to a nominal one
- V3: Feature extraction (fixed-width summary statistics per fragment) and a stratified train/test split
- V4: First evaluated model — a constrained Decision Tree, scored on precision/recall/F1
- V5: Logistic Regression added, both models compared, error modes interpreted
- V6: Final documentation pass and public GitHub release
