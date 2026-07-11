# Satellite Telemetry Anomaly Detection V3 Algorithm

## Goal
Convert variable-length fragments into a fixed-width feature matrix, verify the features carry label signal, and freeze a stratified train/test split. The feature set is deliberately simple summary statistics (info from the V2 plots of what anomalies look like per channel) because the project's point is honest workflow and imbalance-aware evaluation.

## Inputs
Unchanged from V1.

## Outputs
- Everything from V2 (cleaned `fragments_df`; pass-1 EDA figures)
- New: `features_df` (fixed-width feature matrix)
- New: pass-2 EDA figures — per-feature distribution plots (histograms or box plots) split by label
- New: `X_train, X_test, y_train, y_test` arrays with printed class-balance verification for both splits

## Data Structure
- `fragments_df` — Unchanged from V1.
- **New: `features_df`** — fixed-width feature matrix for modeling, 2,123 rows, ~12–15 feature columns:
  - `fragment_id` (join key back to fragments)
  - `mean`, `std`, `min`, `max`, `range` : value-level summary stats
  - `n_samples` : fragment length
  - `n_unique`, `duplicate_ratio` : value-uniqueness stats (flatlined/stuck channels show up here)
  - `diff_mean_abs`, `diff_std` : first-difference statistics (spikes and noise-character changes show up here)
  - `channel` : encoded categorical
  - `label` : target

## Core Functions

1. **`load_fragments(data_dir) -> fragments_df`** — Unchanged from V1.

2. **`clean_fragments(fragments_df) -> fragments_df`** — Unchanged from V1.

3. **`extract_features(fragments_df) -> features_df`**
   - *Input:* cleaned `fragments_df`.
   - *Output:* `features_df` as described under Data Structure.
   - *Process:* For each fragment, compute the summary statistics from its NumPy value array (level stats, length, uniqueness stats, first-difference stats via `np.diff`). Encode `channel` as a categorical. Assert no NaNs in the output and row count still 2,123.
   - *Purpose:* Fixed-width representation so standard sklearn classifiers apply. Each feature maps to an anomaly mode seen in V2 (level shift → mean/range; flatline → duplicate_ratio/n_unique; spike/noise change → diff stats).

4. **`run_eda(fragments_df, features_df) -> saved figures` (extended: pass 2)**
   - Pass 1 (raw fragments) — Unchanged from V2.
   - *New pass 2 — Input:* `features_df`. *Output:* per-feature distribution figures split by label, saved to `figures/`. *Process:* for each feature column, plot the anomalous vs. nominal distributions on shared axes. *Purpose:* confirm before modeling that at least some features visibly separate the classes : if none do, that's a finding to address now, not future versions

5. **`split_data(features_df) -> X_train, X_test, y_train, y_test`**
   - *Input:* `features_df`.
   - *Output:* train/test feature matrices and label vectors.
   - *Process:* `train_test_split` with `stratify=y`, `test_size≈0.2`, fixed `random_state`. Then verify: print class balance of both splits and assert both are within a small tolerance of the overall ~20% positive rate.
   - *Purpose:* Under 80/20 imbalance, an unlucky unstratified split could leave the test set with too few anomalies to measure recall meaningfully. Stratification plus a fixed seed makes every later result reproducible and comparable across V4/V5.

## Execution Flow
1. `load_fragments` → 2. `clean_fragments` → 3. `run_eda` (pass 1, raw fragments) → 4. `extract_features` → 5. `run_eda` (pass 2, features) → 6. `split_data`.

## Design Rule
Fixed-width summary statistics. The feature list is the ~12–15 columns above, each justified by a V2 observation; anything without that justification stays out. If this version's own pass-2 EDA (Core Function 4) finds a feature that doesn't separate the classes, expansion is allowed.
