# Runs both models on the test set and produces basically everything in
# results/ and the confusion matrix / PR curve / autopsy figures.
# Biggest file in the project, kept it as one function since all these steps
# share the same predictions and I didn't want to pass y_pred around everywhere.
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
        # logistic regression was trained on one-hot encoded channels, so the
        # test set needs the same treatment before predict() will work on it
        if model_name == 'logistic_regression':
            X_test_model = pd.get_dummies(X_test, columns=['channel'])
            # reindex here in case the test set doesn't have every channel
            # value that the training set had, otherwise column mismatch error
            X_test_model = X_test_model.reindex(columns=model.feature_names_in_, fill_value=0)
        else:
            X_test_model = X_test

        y_pred = model.predict(X_test_model)

        # this is for the autopsy plots below, need the raw fragment (not just
        # the feature row) to actually plot the value curve
        test_fragments = fragments_df.loc[X_test.index]

        missed = test_fragments[(y_test.values == 1) & (y_pred == 0)]   # false negatives
        caught = test_fragments[(y_test.values == 1) & (y_pred == 1)]   # true positives
        print(f'{model_name}: {len(missed)} missed anomalies, {len(caught)} correctly caught')

        # for each channel, plot the missed anomalies next to the caught ones
        # so I can actually see what the model is getting wrong, not just a number
        for channel_name, channel_missed in missed.groupby('channel'):
            channel_caught = caught[caught['channel'] == channel_name]
            if len(channel_caught) == 0:
                continue   # nothing to compare against for this channel

            plot_fragment_comparison(
                channel_missed, channel_caught, 'Missed', 'Caught', channel_name,
                f'figures/autopsy_{model_name}_{channel_name}_missed_vs_caught.png'
    )

        # anomalous class (label 1) is what actually matters here, accuracy
        # alone would be misleading since ~80% of fragments are nominal anyway
        report = classification_report(y_test, y_pred)
        print(f'------------------- {model_name} -------------------')
        print(report)

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Nominal', 'Anomalous'])
        disp.plot()
        plt.title(f'{model_name} Confusion Matrix')
        plt.savefig(f'figures/confusion_matrix_{model_name}.png')
        plt.close()

        # printing accuracy too but only as a footnote, comparing it against
        # what you'd get by just always guessing "nominal" for everything
        accuracy = accuracy_score(y_test, y_pred)
        baseline_accuracy = y_test.value_counts().max() / len(y_test)
        print(f'(footnote) {model_name} accuracy: {accuracy:.2%} vs. always-predicted-baseline accuracy: {baseline_accuracy:.2%}')

        # only the tree has .feature_importances_, logistic regression
        # doesn't expose it the same way so skipping that for LR
        if model_name == 'decision_tree':
            importances = model.feature_importances_
            feature_names = X_test.columns
            importance_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)

        report_dict = classification_report(y_test, y_pred, output_dict=True)

        # logging every run to metrics_summary.csv so I have a record of
        # every model I've ever trained, not just the final numbers
        row = {
            'model': model_name,
            'hyperparameters': model.max_depth if model_name == 'decision_tree' else 'class_weight=balanced',
            'timestamp': str(datetime.now()),
            'precision': report_dict['1']['precision'],
            'recall': report_dict['1']['recall'],
            'f1': report_dict['1']['f1-score'],
        }

        os.makedirs('results', exist_ok=True)

        # same info as the row above but saved as its own json file per run,
        # kind of redundant with metrics_summary.csv but nice to have the
        # full detail per run instead of just one row in a csv
        log_entry = {
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

        # mode='a' so this appends instead of overwriting, header only gets
        # written the very first time the file is created
        metrics_path = 'results/metrics_summary.csv'
        pd.DataFrame([row]).to_csv(metrics_path, mode='a', header=not os.path.exists(metrics_path), index=False)

        # precision_recall_curve needs probabilities, not just the 0/1 prediction
        y_proba = model.predict_proba(X_test_model)[:,1]
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

    # finishing the shared PR curve figure now that both models have been added to it
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.savefig('figures/pr_curves.png')
    plt.close()

    print('\n--- Model Comparison ---')
    for model_name, metrics in comparison.items():
        print(f'{model_name} :  precision={metrics["precision"]:.2f}, recall={metrics["recall"]:.2f}, f1={metrics["f1"]:.2f}')
