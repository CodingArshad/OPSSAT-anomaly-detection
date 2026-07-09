# Entry point: will run load_fragments -> clean_fragments and print a summary.
# See docs/algorithm.md (V1) for the intended execution flow.

from src.load import load_fragments
from src.clean import clean_fragments
from src.eda import run_eda

dataset = load_fragments(data_dir='data/raw')
clean_df = clean_fragments(dataset)

run_eda(clean_df)
    
print(f'Final dataset has {len(clean_df)} fragments with {clean_df["channel"].nunique()} channels and an anomaly rate of {clean_df["label"].mean():.2%}')
print(clean_df['channel'].value_counts())
