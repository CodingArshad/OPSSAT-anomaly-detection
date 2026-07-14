# Loads the raw OPSSAT-AD files and turns them into one row-per-fragment DataFrame
import pandas as pd
import numpy as np

def load_fragments(data_dir='data/raw'):
    # One row per fragment
    dataset = pd.read_csv(data_dir + '/dataset.csv')
    # One row per sample, way more rows
    segments = pd.read_csv(data_dir + '/segments.csv')

    # Need time order before making arrays
    sorted_segments = segments.sort_values('timestamp')
    grouped = sorted_segments.groupby('segment')

    # One array per fragment now
    values = grouped['value'].apply(lambda group: group.to_numpy())
    timestamps = grouped['timestamp'].apply(lambda group: group.to_numpy())

    # Match the names I use elsewhere
    dataset.rename(columns={'segment': 'fragment_id', 'anomaly': 'label'}, inplace=True)

    # Don't need the other columns
    dataset = dataset[['fragment_id', 'label', 'channel']]

    # So values/timestamps line up with the right row
    dataset.set_index('fragment_id', inplace=True)
    dataset['values'] = values
    dataset['timestamps'] = timestamps
    dataset.reset_index(inplace=True)

    # Published fact about this dataset
    assert len(dataset) == 2123, 'Expected 2123 fragments'
    assert dataset['channel'].nunique() == 9, 'Expected 9 channels'
    assert dataset['label'].mean() >= 0.15 and dataset['label'].mean() <= 0.25, 'Expected anomaly rate between 15% and 25%'
    print(f'Loaded {len(dataset)} fragments with {dataset["channel"].nunique()} channels and an anomaly rate of {dataset["label"].mean():.2%}')

    return dataset
