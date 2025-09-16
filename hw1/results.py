import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def show_results(report, cm):
    """
    Print metrics and plot confusion matrix.
    Parameters:
    report (dict): Classification report as a dictionary.
    cm (array): Confusion matrix.
    """
    
    print("Classification Report:")
    for label, metrics in report.items():
        print(label, metrics)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"])
    plt.title("Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.show()
