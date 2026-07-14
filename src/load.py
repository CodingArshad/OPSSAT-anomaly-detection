# loads the raw OPSSAT-AD files and turns them into one row-per-fragment DataFrame
import pandas as pd
import numpy as np

def load_fragments(data_dir='data/raw'):
    dataset = pd.read_csv(data_dir + '/dataset.csv')  # one row per fragment
    segments = pd.read_csv(data_dir + '/segments.csv')  # one row per sample, way more rows

    sorted_segments = segments.sort_values('timestamp')  # need time order before making arrays
    grouped = sorted_segments.groupby('segment')

    values = grouped['value'].apply(lambda group: group.to_numpy())  # one array per fragment now
    timestamps = grouped['timestamp'].apply(lambda group: group.to_numpy())

    dataset.rename(columns={'segment': 'fragment_id', 'anomaly': 'label'}, inplace=True)  # match the names I use elsewhere

    dataset = dataset[['fragment_id', 'label', 'channel']]  # don't need the other columns

    dataset.set_index('fragment_id', inplace=True)  # so values/timestamps line up with the right row
    dataset['values'] = values
    dataset['timestamps'] = timestamps
    dataset.reset_index(inplace=True)

    assert len(dataset) == 2123, 'Expected 2123 fragments'  # published fact about this dataset
    assert dataset['channel'].nunique() == 9, 'Expected 9 channels'
    assert dataset['label'].mean() >= 0.15 and dataset['label'].mean() <= 0.25, 'Expected anomaly rate between 15% and 25%'
    print(f'Loaded {len(dataset)} fragments with {dataset["channel"].nunique()} channels and an anomaly rate of {dataset["label"].mean():.2%}')

    return dataset
