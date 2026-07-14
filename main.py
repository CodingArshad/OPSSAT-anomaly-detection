# Entry point for the whole project. Runs the full pipeline start to finish:
# load -> clean -> feature extraction -> EDA -> split -> train both models -> evaluate.
# Full spec / reasoning for each step is in docs/algorithm.md.

from src.load import load_fragments
from src.clean import clean_fragments
from src.eda import run_eda
from src.features import extract_features
from src.split import split_data
from src.models import train_models
from src.evaluate import evaluate_models

print('=== Loading data ===')
dataset = load_fragments(data_dir='data/raw')
clean_df = clean_fragments(dataset)

print('\n=== Extracting features ===')
features_df = extract_features(clean_df)

print('\n=== Running EDA (see figures/) ===')
run_eda(clean_df, features_df)

print('\n=== Splitting into train/test ===')
X_train, X_test, y_train, y_test = split_data(features_df)

print('\n=== Training models ===')
models = train_models(X_train, y_train)

print('\n=== Evaluating models ===')
evaluate_models(models, X_test, y_test, clean_df)

print('\nDone. Figures in figures/, models + metrics in results/.')
