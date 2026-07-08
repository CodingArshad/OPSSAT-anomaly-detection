# Satellite Telemetry Anomaly Detection V1 Algorithm

## Goal
Load the OPSSAT-AD dataset into an analysis-ready DataFrame and verify its integrity. V1 succeeds when the data is parsed into exactly 2,123 labeled fragments across 9 channels at a ~20% anomaly rate, with data-quality issues (NaNs, duplicates, non-numeric values, empty fragments) either resolved or explicitly logged. This is the foundation for the full project goal: a binary classifier (anomalous fragment vs. not) evaluated honestly under ~80/20 class imbalance.

---

## Inputs
Source: OPSSAT-AD dataset, Zenodo record 12588359 (https://zenodo.org/records/12588359). Paper: arXiv:2407.04730. Reference code (sanity-check only, not for copying): kplabs-pl/OPS-SAT-AD.

Data:
- `dataset.csv` (2,123 rows + header) — one row per fragment: `segment` (fragment ID), `anomaly` (0/1 label), `train` (the reference repo's own train/test split — unused here, V3 does its own stratified split), `channel`, plus ~19 precomputed summary-statistic columns (`mean`, `var`, `std`, `kurtosis`, `skew`, peak counts, diff stats, etc.) that belong to a later feature-extraction version, not V1.
- `segments.csv` (303,493 rows + header) — one row per raw sample: `channel`, `timestamp`, `value`, `label` (a four-value categorical tag — `anomaly`/`a2`/`a3`/`a4` — present on every row regardless of the binary `anomaly` flag, so it is *not* the anomaly indicator and is ignored), `sampling`, `anomaly` (0/1, matches `dataset.csv`), `segment` (fragment ID, joins to `dataset.csv.segment`), `train`.

Every `segment` value in `dataset.csv` has a matching group of rows in `segments.csv` (verified 1:1, no orphans). Fragment lengths range 8–1,040 samples.

## Outputs
- `fragments_df`: cleaned, verified per-fragment DataFrame 
- Report of the cleaning audit: what was checked, what was found, what was done about it

## Data Structure
`fragments_df` — raw per-fragment table, 2,123 rows:
- `fragment_id`: unique identifier (from `dataset.csv`'s `segment` column)
- `channel`: which of the 9 telemetry channels this fragment came from
- `values`: the fragment's raw telemetry samples as a NumPy ndarray, built by grouping `segments.csv` rows by `segment` and sorting by `timestamp` (fragments are variable-length, so values stay as arrays rather than wide columns)
- `timestamps`: sample timestamps for the fragment, same grouping
- `label`: 1 = anomalous, 0 = nominal (from `dataset.csv`'s `anomaly` column)

## Core Functions

1. **`load_fragments(data_dir) -> fragments_df`**
   - *Input:* path to the directory containing `dataset.csv` and `segments.csv`.
   - *Output:* `fragments_df` as described above.
   - *Process:* Read both CSVs. Group `segments.csv` by `segment`, sorted by `timestamp`, into per-fragment `values`/`timestamps` arrays. Take `fragment_id`, `channel`, `label` from `dataset.csv`. Merge on fragment ID. Then assert the published dataset facts: exactly 2,123 fragments, exactly 9 channels, positive rate within [0.15, 0.25]. A failed assertion means the loader misread the layout — fail loudly, don't proceed.
   - *Purpose:* One trusted entry point for the data. The assertions are the version's "it works" signal: if they pass, everything downstream can trust the fragment table.

2. **`clean_fragments(fragments_df) -> fragments_df`**
   - *Input:* raw `fragments_df` from `load_fragments`.
   - *Output:* cleaned `fragments_df`.
   - *Process:* Check for duplicate `fragment_id`s, empty (zero-length) fragments, and NaN/non-numeric values inside value arrays — drop any row that fails a check, and print a count for each check whether or not it found anything. Direct inspection of the real archive found it already clean (no NaNs, no duplicates, no empty fragments); the audit runs anyway and logs that finding, since the point is a visible, repeatable check — not an assumption that the data stays clean.
   - *Purpose:* Guarantee the fragment table is analysis-ready, with an auditable record of what was checked.

## Execution Flow
1. `load_fragments` → 2. `clean_fragments` → print summary (fragment count, per-channel counts, anomaly rate, cleaning log).
