from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from datetime import datetime
import matplotlib.pyplot as plt
import os
import pandas as pd
import json

def evaluate_models(models, X_test, y_test):
    for model_name, model in models.items():
        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred)
        print(f'------------------- {model_name} -------------------')
        print(report)

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Nominal', 'Anomalous'])
        disp.plot()
        plt.title(f'{model_name} Confusion Matrix')
        plt.savefig(f'figures/confusion_matrix_{model_name}.png')
        plt.close()

        accuracy = accuracy_score(y_test, y_pred)
        baseline_accuracy = y_test.value_counts().max() / len(y_test)
        print(f'(footnote) {model_name} accuracy: {accuracy:.2%} vs. always-predicted-baseline accuracy: {baseline_accuracy:.2%}')

        importances = model.feature_importances_
        feature_names = X_test.columns  
        importance_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)

        report_dict = classification_report(y_test, y_pred, output_dict=True)

        row = {
            'model': model_name,
            'max_depth': model.max_depth,
            'timestamp': str(datetime.now()),
            'precision': report_dict['1']['precision'],
            'recall': report_dict['1']['recall'],
            'f1': report_dict['1']['f1-score'],
        }

        os.makedirs('results', exist_ok=True)
        
        log_entry = {
            'model': model_name,
            'max_depth': model.max_depth,
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
        pd.DataFrame([row]).to_csv(metrics_path, mode='a', header=not os.path.exists(metrics_path), index=False)


        plt.figure(figsize=(10, 5))
        importance_series.plot(kind='bar')
        plt.title(f'{model_name} Feature Importance')
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'figures/feature_importance_{model_name}.png')
        plt.close()

        print(importance_series.head(3))  # Print top 3 important features
