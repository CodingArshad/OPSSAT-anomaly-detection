# Satellite Telemetry Anomaly Detection V2 Algorithm

## Goal
Build visuals for the raw data before modeling. V2 answers three questions with saved figures: How are fragments distributed across the 9 channels? How does the anomaly rate vary by channel? and What does an anomalous fragment look like next to a nominal one from the same channel?

## Inputs
Unchanged from V1.

## Outputs
- Everything from V1 (verified, cleaned `fragments_df`; cleaning audit log)
- New: saved EDA figures in `figures/` —
  - `channel_counts_anomaly_rates.png`: bar chart of fragment counts per channel, annotated with per-channel anomaly rate
  - `fragments_ch{N}_anomalous_vs_nominal.png` (one per channel): line plots of anomalous fragments overlaid or side-by-side with nominal fragments from the same channel

## Data Structure
Unchanged from V1

## Core Functions

1. **`load_fragments(data_dir) -> fragments_df`** — Unchanged from V1.

2. **`clean_fragments(fragments_df) -> fragments_df`** — Unchanged from V1.

3. **`run_eda(fragments_df) -> saved figures` (pass 1)**
   - *Input:* cleaned `fragments_df`.
   - *Output:* figures saved to `figures/` (listed under Outputs).
   - *Process:* 
      (a) Group by channel; plot fragment counts and anomaly rates per channel. 
      (b) For each channel, sample a handful of anomalous and nominal fragments and plot their raw value series against their timestamps(with labels). Keep the same axis scales within a channel so differences are not artifacts of scaling. Note observations inline (as figure captions or a short markdown notes file): Do anomalies look like spikes, level shifts, flatlines, or noise-character changes, and does this differ by channel?
   - *Purpose:* Ground the V3 feature choices in what the anomalies actually look like, rather than picking summary statistics blind. Also establishes whether class imbalance is uniform across channels, which matters for interpreting per-channel performance later.

## Execution Flow
1. `load_fragments` → 2. `clean_fragments` → 3. `run_eda` (pass 1, on raw fragments).

## Design Rule
EDA is for visualization, not modeling. V2 produces exactly the two figure families above, no clustering, no statistical tests, etc... The version is successful when the figures exist and can answer, per channel, "what does an anomaly look like here?" If a figure doesn't help answer that, it doesn't get made.
