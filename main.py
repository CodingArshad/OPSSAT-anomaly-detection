# Entry point: will run load_fragments -> clean_fragments and print a summary.
# See docs/algorithm.md (V1) for the intended execution flow.

from src.load import load_fragments
from src.clean import clean_fragments
from src.eda import run_eda
from src.features import extract_features
from src.split import split_data
from src.models import train_models
from src.evaluate import evaluate_models

dataset = load_fragments(data_dir='data/raw')
clean_df = clean_fragments(dataset)
features_df = extract_features(clean_df)
X_train, X_test, y_train, y_test = split_data(features_df)

models = train_models(X_train, y_train)
run_eda(clean_df, features_df)
evaluate_models(models, X_test, y_test)

print(features_df.head())
print(f'Final dataset has {len(clean_df)} fragments with {clean_df["channel"].nunique()} channels and an anomaly rate of {clean_df["label"].mean():.2%}')
print(clean_df['channel'].value_counts())
