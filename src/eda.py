# Makes all the EDA figures. Gets called twice: once on the raw fragments
# (pass 1) and once on the extracted features (pass 2). Also has the plotting
# function the misclassification autopsy reuses later in evaluate.py.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_eda(fragments_df, features_df):
    # how many fragments per channel, and what fraction of each channel is anomalous
    groups = fragments_df.groupby('channel')
    count = groups.size()
    anomaly_rate = groups['label'].mean()

    plt.figure(figsize=(10, 5))
    plt.bar(count.index, count.values)
    plt.title('Fragment Counts and Anomaly Rates by Channel')
    plt.xlabel('Channel')
    plt.ylabel('Fragment Count')
    # putting the anomaly rate as text above each bar instead of a separate chart
    for channel, cnt, rate in zip(count.index, count.values, anomaly_rate.values):
        plt.text(channel, cnt, f'{rate:.2%}', ha='center', va='bottom')
    plt.savefig('figures/channel_counts_anomaly_rates.png')
    plt.close()

    # one plot per channel showing what an anomalous fragment actually looks
    # like next to a nominal one, this is the plot that first showed me
    # anomalies are basically a smooth curve with a glitch cut into it
    for group_name, channel_df in groups:
        anomalous = channel_df[channel_df['label'] == 1]
        nominal = channel_df[channel_df['label'] == 0]
        plot_fragment_comparison(anomalous, nominal, 'Anomalous', 'Nominal', group_name, f'figures/fragments_{group_name}_anomalous_vs_nominal.png')

    # pass 2 only happens once features_df exists, so this part gets skipped
    # the first time run_eda is called (fragments_df has no feature columns yet)
    feature_columns = [col for col in features_df.columns if col != 'fragment_id' and col != 'label' and col != 'channel']

    for feature in feature_columns:
        anomalous = features_df[features_df['label'] == 1]
        nominal = features_df[features_df['label'] == 0]

        plt.figure(figsize=(10, 5))

        # using the SAME bins for both histograms on purpose, if I let
        # matplotlib pick bins separately for each one it can make a fake
        # looking gap between the two distributions that isn't actually there
        feature_min = np.min(features_df[feature])
        feature_max = np.max(features_df[feature])
        bins = np.linspace(feature_min, feature_max, 30)

        plt.hist(anomalous[feature], alpha=0.5, label='Anomalous', color='red', density=True, bins=bins)
        plt.hist(nominal[feature], alpha=0.5, label='Nominal', color='blue', density=True, bins=bins)

        plt.title(f'{feature} Distribution: Anomalous vs Nominal')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(f'figures/feature_{feature}_anomalous_vs_nominal.png')
        plt.close()

def plot_fragment_comparison(group_a_df, group_b_df, label_a, label_b, channel_name, filename):
    # only plotting up to 5 fragments per side, plotting all of them would
    # just be a wall of overlapping lines and impossible to read
    sample_a = group_a_df.sample(n=min(5, len(group_a_df)), random_state=1)
    sample_b = group_b_df.sample(n=min(5, len(group_b_df)), random_state=1)

    plt.figure(figsize=(10, 5))

    # converting timestamps to "seconds since fragment started" so fragments
    # from totally different times can still be compared on the same x-axis
    for i, (idx, row) in enumerate(sample_a.iterrows()):
        timestamps = pd.to_datetime(row['timestamps'])
        elapsed_seconds = (timestamps - timestamps[0]).total_seconds()
        # only labeling the first line of each color so the legend doesn't
        # end up with 5 identical "Anomalous" entries
        plt.plot(elapsed_seconds, row['values'], label=label_a if i == 0 else None, color='red')

    for i, (idx, row) in enumerate(sample_b.iterrows()):
        timestamps = pd.to_datetime(row['timestamps'])
        elapsed_seconds = (timestamps - timestamps[0]).total_seconds()
        plt.plot(elapsed_seconds, row['values'], label=label_b if i == 0 else None, color='blue')

    plt.title(f'{label_a} vs {label_b} Fragments on Channel {channel_name}')
    plt.xlabel('Elapsed Time (s)')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(filename)
    plt.close()
