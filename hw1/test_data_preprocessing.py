import pytest
from data_preprocessing import read_data, data_find_duplicates, data_preprocessing, train_test_split_data
import pandas as pd
import numpy as np

class TestDataPreprocessing():
    def test_read_data(self):
        '''Test reading data from a CSV file.'''
        file_path = 'data/raw/CreditCardFraud_updated.csv'  # Replace with your actual file path
        data = read_data(file_path)
        assert data is not None
        assert not data.empty

    def test_data_find_duplicates(self):
        '''Test finding duplicate columns in the data.'''
        file_path = 'data/raw/CreditCardFraud_updated.csv'  # Replace with your actual file path
        data = read_data(file_path)
        # This function prints output; we can only check that it runs without error
        data_find_duplicates(data)

    def test_data_preprocessing(self):
        '''Test data preprocessing steps.'''
        file_path = 'data/raw/CreditCardFraud_updated.csv'  # Replace with your actual file path
        data = read_data(file_path)
        preprocessed_data = data_preprocessing(data)
        assert preprocessed_data is not None
        assert not preprocessed_data.empty

    def test_train_test_split_data(self):
        '''Test train-test split functionality.'''
        file_path = 'data/raw/CreditCardFraud_updated.csv'  # Replace with your actual file path
        data = read_data(file_path)
        test_X_train, test_X_test, test_y_train, test_y_test = train_test_split_data(data, target_column='isFraud', test_size=0.3, random_state=42)
        assert test_X_train is not None
        assert test_X_test is not None
        assert test_y_train is not None
        assert test_y_test is not None