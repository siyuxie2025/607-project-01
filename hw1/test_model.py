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
        cols = ['creditLimit', 'currentBalance', 'availableMoney', 'transactionAmount']
        data = pd.DataFrame({
            'creditLimit': [1000, 2000, 1500, 3000, 2500, 1200, 1800, 2200],
            'currentBalance': [500, 1500, 800, 2500, 2000, 600, 1200, 1800],
            'availableMoney': [400, 1300, 700, 2400, 1900, 500, 1100, 1700],
            'transactionAmount': [100, 200, 150, 300, 250, 120, 180, 220],
            'isFraud': [0, 1, 0, 1, 1, 0, 1, 0]
        })

        X = data[['creditLimit', 'currentBalance', 'availableMoney', 'transactionAmount']]
        y = data['isFraud']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split_data(data,
                                                                  target_column='isFraud', 
                                                                  test_size=0.5, random_state=42)

        # Run analysis
        model, report, cm = run_analysis(X_train, X_test, y_train, y_test)

        # Check that the model is a LogisticRegression instance
        assert isinstance(model, LogisticRegression)

        # Check that the report is a dictionary
        assert isinstance(report, dict)

        # Check that the confusion matrix is a numpy array
        assert isinstance(cm, np.ndarray)