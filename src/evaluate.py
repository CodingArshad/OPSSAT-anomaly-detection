# runs both models on the test set, makes the confusion matrix / PR curve / autopsy figures
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_recall_curve, average_precision_score
from datetime import datetime
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
from src.eda import plot_fragment_comparison

def evaluate_models(models, X_test, y_test, fragments_df):
    comparison = {}
    pr_fig = plt.figure(figsize=(8,6))  # both models' PR curves go on this one figure

    for model_name, model in models.items():
        if model_name == 'logistic_regression':
            X_test_model = pd.get_dummies(X_test, columns=['channel'])  # match the one-hot encoding used in training
            X_test_model = X_test_model.reindex(columns=model.feature_names_in_, fill_value=0)  # test set might be missing a channel value
        else:
            X_test_model = X_test

        y_pred = model.predict(X_test_model)

        test_fragments = fragments_df.loc[X_test.index]  # need the raw fragment for the autopsy plots below

        missed = test_fragments[(y_test.values == 1) & (y_pred == 0)]   # false negatives
        caught = test_fragments[(y_test.values == 1) & (y_pred == 1)]   # true positives
        print(f'{model_name}: {len(missed)} missed anomalies, {len(caught)} correctly caught')

        for channel_name, channel_missed in missed.groupby('channel'):  # plot missed vs caught per channel
            channel_caught = caught[caught['channel'] == channel_name]
            if len(channel_caught) == 0:
                continue   # nothing to compare against for this channel

            plot_fragment_comparison(
                channel_missed, channel_caught, 'Missed', 'Caught', channel_name,
                f'figures/autopsy_{model_name}_{channel_name}_missed_vs_caught.png'
    )

        report = classification_report(y_test, y_pred)  # anomalous class is what matters, not raw accuracy
        print(f'------------------- {model_name} -------------------')
        print(report)

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Nominal', 'Anomalous'])
        disp.plot()
        plt.title(f'{model_name} Confusion Matrix')
        plt.savefig(f'figures/confusion_matrix_{model_name}.png')
        plt.close()

        accuracy = accuracy_score(y_test, y_pred)
        baseline_accuracy = y_test.value_counts().max() / len(y_test)  # what you'd get just guessing nominal every time
        print(f'(footnote) {model_name} accuracy: {accuracy:.2%} vs. always-predicted-baseline accuracy: {baseline_accuracy:.2%}')

        if model_name == 'decision_tree':  # only the tree exposes feature_importances_
            importances = model.feature_importances_
            feature_names = X_test.columns
            importance_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)

        report_dict = classification_report(y_test, y_pred, output_dict=True)

        row = {  # one row per run, appended to metrics_summary.csv below
            'model': model_name,
            'hyperparameters': model.max_depth if model_name == 'decision_tree' else 'class_weight=balanced',
            'timestamp': str(datetime.now()),
            'precision': report_dict['1']['precision'],
            'recall': report_dict['1']['recall'],
            'f1': report_dict['1']['f1-score'],
        }

        os.makedirs('results', exist_ok=True)

        log_entry = {  # same info as row but as its own json, more detail than the csv keeps
            'model': model_name,
            'hyperparameters': model.max_depth if model_name == 'decision_tree' else 'class_weight=balanced',
            'timestamp': str(datetime.now()),
            'metrics': {
                'precision': report_dict['1']['precision'],
                'recall': report_dict['1']['recall'],
                'f1': report_dict['1']['f1-score'],
                'accuracy': accuracy,
                'baseline_accuracy': baseline_accuracy,
            },
        }

        os.makedirs('results/experiment_log', exist_ok=True)
        log_filename = f'results/experiment_log/{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(log_filename, 'w') as f:
            json.dump(log_entry, f, indent=2)

        metrics_path = 'results/metrics_summary.csv'
        pd.DataFrame([row]).to_csv(metrics_path, mode='a', header=not os.path.exists(metrics_path), index=False)  # append, header only on first write

        y_proba = model.predict_proba(X_test_model)[:,1]  # PR curve needs probabilities, not just 0/1
        precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)
        plt.figure(pr_fig.number)
        plt.plot(recall_curve, precision_curve, label=f'{model_name} (AP={avg_precision:.2f})')

        if model_name == 'decision_tree':
            plt.figure(figsize=(10, 5))
            importance_series.plot(kind='bar')
            plt.title(f'{model_name} Feature Importance')
            plt.xlabel('Feature')
            plt.ylabel('Importance')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'figures/feature_importance_{model_name}.png')
            plt.close()
            print(importance_series.head(3))   # just want to see the top 3 at a glance

        comparison[model_name] = {
            'precision' : report_dict['1']['precision'],
            'recall' : report_dict['1']['recall'],
            'f1' : report_dict['1']['f1-score']
        }

    plt.xlabel('Recall')  # wrapping up the shared PR curve figure now
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.savefig('figures/pr_curves.png')
    plt.close()

    print('\n--- Model Comparison ---')
    for model_name, metrics in comparison.items():
        print(f'{model_name} :  precision={metrics["precision"]:.2f}, recall={metrics["recall"]:.2f}, f1={metrics["f1"]:.2f}')
