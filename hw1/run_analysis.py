from data_preprocessing import read_data, data_find_duplicates, data_preprocessing, train_test_split_data
from EDA import eda_features, eda_cvv_match, eda_transaction_amount, eda_categorical_features
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from model import run_analysis
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == "__main__":

    # Example usage
    file_path = 'data/raw/CreditCardFraud_updated.csv'  # Replace with your actual file path
    data = read_data(file_path)
    data_find_duplicates(data)

    preprocessed_data = data_preprocessing(data)
    print("Preprocessing completed.")
    
    #save preprocessed data
    preprocessed_data.to_csv('data/processed/preprocessed_data.csv', index=False)
    print("Preprocessed data saved to data/processed/preprocessed_data.csv")

    # EDA part in EDA.py
    eda_features(preprocessed_data)
    print("EDA on features completed.")
    eda_cvv_match(preprocessed_data)
    print("EDA on CVV match completed.")
    eda_transaction_amount(preprocessed_data)
    print("EDA on transaction amount completed.")
    eda_categorical_features(preprocessed_data)
    print("EDA on categorical features completed.")

    X_train, X_test, y_train, y_test = train_test_split_data(preprocessed_data, 
                                                             target_column='isFraud')
    print("Train-test split completed.")
    
    run_analysis(X_train, X_test, y_train, y_test)
    print("Model training and evaluation completed.")