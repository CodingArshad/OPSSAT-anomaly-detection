# V2 EDA Observations

Notes on `channel_counts_anomaly_rates.png` and the 9 `fragments_{channel}_anomalous_vs_nominal.png` figures.

## Cross-channel pattern

Nominal fragments trace smooth, roughly bell-shaped curves, likely tracking something physically continuous, such as the satellite's orientation relative to the sun/Earth cycling once per orbit.

Anomalous fragments largely follow that same underlying smooth shape, but have sharp, localized
interruptions cut into it: sudden spikes, vertical level shifts, or flatlines that break an otherwise continuous curve before it resumes. This shows up across nearly every channel (clearest in CADC0873, CADC0888, CADC0890) and is the main visual signature of an anomaly here(not a totally different shape, but a smooth signal with a glitch layered on top)

This suggests V3 features sensitive to *local* discontinuity (e.g. large sample-to-sample differences, second-derivative magnitude, peak/spike counts) are more likely to separate the classes than global summary stats like plain mean or variance, which would wash out a brief, sharp interruption.

## Channel-specific notes

- **CADC0886**: nominal fragments peak around 0.6–0.7; anomalous fragments cap out around 0.25–0.4.
  A visible amplitude reduction, not just a shape distortion — anomalies here look "damped" relative to nominal behavior.
- **CADC0872**: fragments mostly overlap in duration (~0-200s elapsed), but one sampled anomalous fragment runs much longer (~900s) than any other fragment shown. Worth remembering that fragment duration itself varies a lot (8-1,040 samples per the raw archive) and isn't controlled for in these plots.

## Anomaly rate is not uniform across channels

Per `channel_counts_anomaly_rates.png`: rates range from 0.00% (CADC0884) to 78.57% (CADC0890), with most channels landing in the 15-35% range. CADC0886 and CADC0890 have very few total fragments (11 and 14 respectively), so their extreme rates should be read with that small-sample caveat in mind; a couple of mislabeled or unusual fragments would swing the percentage a lot more there than on a channel with a large number of fragments.