"""Save figures to the `figures` directory. One chart with per-channel fragment counts and anomaly rates. Nine charts with anomalous fragments plotted alongside nominal ones from the same channel(raw value vs. timestamp, same axis scale within each channel)"""
import pandas as pd

import matplotlib.pyplot as plt

def run_eda(fragments_df):
    groups = fragments_df.groupby('channel')
    count = groups.size()
    anomaly_rate = groups['label'].mean()

    plt.figure(figsize=(10, 5))
    plt.bar(count.index, count.values)
    plt.title('Fragment Counts and Anomaly Rates by Channel')
    plt.xlabel('Channel')
    plt.ylabel('Fragment Count')
    for channel, cnt, rate in zip(count.index, count.values, anomaly_rate.values):
        plt.text(channel, cnt, f'{rate:.2%}', ha='center', va='bottom')
    plt.savefig('figures/channel_counts_anomaly_rates.png')
    plt.close()

    for group_name, channel_df in groups:
        anomalous = channel_df[channel_df['label'] == 1]
        nominal = channel_df[channel_df['label'] == 0]

        anomalous_sample = anomalous.sample(n=min(5, len(anomalous)), random_state=1)
        nominal_sample = nominal.sample(n=min(5, len(nominal)), random_state=1)

        plt.figure(figsize=(10, 5))

        for i, (idx, row) in enumerate(anomalous_sample.iterrows()):
            timestamps = pd.to_datetime(row['timestamps'])   # parse strings into real datetimes
            elapsed_seconds = (timestamps - timestamps[0]).total_seconds()

            plt.plot(elapsed_seconds, row['values'], label='Anomalous' if i == 0 else None, color='red')

        for i, (idx, row) in enumerate(nominal_sample.iterrows()):
            timestamps = pd.to_datetime(row['timestamps'])
            elapsed_seconds = (timestamps - timestamps[0]).total_seconds()
            plt.plot(elapsed_seconds, row['values'], label='Nominal' if i == 0 else None, color='blue')

        plt.title(f'Anomalous vs Nominal Fragments on Channel {group_name}')
        plt.xlabel('Elapsed Time (s)')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(f'figures/fragments_{group_name}_anomalous_vs_nominal.png')
        plt.close()
