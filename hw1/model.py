import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

def run_analysis(X_train, X_test, y_train, y_test):
    """Train a simple logistic regression and evaluate it."""
    # Logistic Regression (baseline)
    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    pd.DataFrame(report).to_json('results/report/model_results.json')
    pd.to_pickle(model, 'results/report/logistic_regression_model.pkl')

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Fraud", "Fraud"],
                 yticklabels=["Not Fraud", "Fraud"])
    plt.title("Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.savefig('results/figures/confusion_matrix.png')

    return model, report, cm


def show_results(report, cm):
    """
    Print metrics and plot confusion matrix.
    Parameters:
    report (dict): Classification report as a dictionary.
    cm (array): Confusion matrix.
    """

   
    # plt.show()