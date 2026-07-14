# makes all the EDA figures, also has the plotting function the autopsy reuses in evaluate.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_eda(fragments_df, features_df):
    groups = fragments_df.groupby('channel')  # fragment counts + anomaly rate per channel
    count = groups.size()
    anomaly_rate = groups['label'].mean()

    plt.figure(figsize=(10, 5))
    plt.bar(count.index, count.values)
    plt.title('Fragment Counts and Anomaly Rates by Channel')
    plt.xlabel('Channel')
    plt.ylabel('Fragment Count')
    for channel, cnt, rate in zip(count.index, count.values, anomaly_rate.values):  # anomaly rate as text above each bar
        plt.text(channel, cnt, f'{rate:.2%}', ha='center', va='bottom')
    plt.savefig('figures/channel_counts_anomaly_rates.png')
    plt.close()

    for group_name, channel_df in groups:  # this is the plot that first showed me anomalies look like a glitch cut into a smooth curve
        anomalous = channel_df[channel_df['label'] == 1]
        nominal = channel_df[channel_df['label'] == 0]
        plot_fragment_comparison(anomalous, nominal, 'Anomalous', 'Nominal', group_name, f'figures/fragments_{group_name}_anomalous_vs_nominal.png')

    feature_columns = [col for col in features_df.columns if col != 'fragment_id' and col != 'label' and col != 'channel']  # the 10 stats from extract_features

    for feature in feature_columns:
        anomalous = features_df[features_df['label'] == 1]
        nominal = features_df[features_df['label'] == 0]

        plt.figure(figsize=(10, 5))

        feature_min = np.min(features_df[feature])  # same bins for both histograms, otherwise it's not a fair comparison
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
    sample_a = group_a_df.sample(n=min(5, len(group_a_df)), random_state=1)  # only 5 each, more would just be a mess of lines
    sample_b = group_b_df.sample(n=min(5, len(group_b_df)), random_state=1)

    plt.figure(figsize=(10, 5))

    for i, (idx, row) in enumerate(sample_a.iterrows()):  # elapsed seconds so fragments from different times still line up
        timestamps = pd.to_datetime(row['timestamps'])
        elapsed_seconds = (timestamps - timestamps[0]).total_seconds()
        plt.plot(elapsed_seconds, row['values'], label=label_a if i == 0 else None, color='red')  # only label the first line, not all 5

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
