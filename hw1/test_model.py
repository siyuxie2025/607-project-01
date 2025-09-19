from model import run_analysis
import pandas as pd
from data_preprocessing import read_data, data_preprocessing, train_test_split_data
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

class TestModel():
    def test_run_analysis(self):
        '''Test the run_analysis function.'''
        # Create a small synthetic dataset for testing
        data = pd.DataFrame({
            'feature1': [0, 1, 0, 1],
            'feature2': [1, 0, 1, 0],
            'isFraud': [0, 1, 0, 1]
        })

        X = data[['feature1', 'feature2']]
        y = data['isFraud']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split_data(data, target_column='isFraud', test_size=0.5, random_state=42)

        # Run analysis
        model, report, cm = run_analysis(X_train, X_test, y_train, y_test)

        # Check that the model is a LogisticRegression instance
        assert isinstance(model, LogisticRegression)

        # Check that the report is a dictionary
        assert isinstance(report, dict)

        # Check that the confusion matrix is a numpy array
        assert isinstance(cm, np.ndarray)