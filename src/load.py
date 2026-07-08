"""load_fragments: parse the raw OPSSAT-AD Zenodo archive into fragments_df.

See docs/algorithm.md (V1) for the spec: expected columns, published-fact
assertions (2,123 fragments / 9 channels / ~20% anomaly rate), and the
confirmed archive layout (dataset.csv + segments.csv).
"""
import pandas as pd
import numpy as np

def load_fragments(data_dir='data/raw'):
    # Load in the data
    dataset = pd.read_csv(data_dir + '/dataset.csv')
    segments = pd.read_csv(data_dir + '/segments.csv')
    
    # Sort and group the data
    sorted_segments = segments.sort_values('timestamp')
    grouped = sorted_segments.groupby('segment')

    # Create the per-fragment arrays
    values = grouped['value'].apply(lambda group: group.to_numpy())
    timestamps = grouped['timestamp'].apply(lambda group: group.to_numpy())

    # Rename the columns
    dataset.rename(columns={'segment': 'fragment_id', 'anomaly': 'label'}, inplace=True)
    
    # Narrowing down the dataset
    dataset = dataset[['fragment_id', 'label', 'channel']]

    # Merge data into a single DataFrame(on fragment ID)
    dataset.set_index('fragment_id', inplace=True)
    dataset['values'] = values
    dataset['timestamps'] = timestamps
    dataset.reset_index(inplace=True)

    # Assert the expected number of fragments and channels
    assert len(dataset) == 2123, 'Expected 2123 fragments'
    assert dataset['channel'].nunique() == 9, 'Expected 9 channels'
    assert dataset['label'].mean() >= 0.15 and dataset['label'].mean() <= 0.25, 'Expected anomaly rate between 15% and 25%'
    print(f'Loaded {len(dataset)} fragments with {dataset["channel"].nunique()} channels and an anomaly rate of {dataset["label"].mean():.2%}')
    
    return dataset