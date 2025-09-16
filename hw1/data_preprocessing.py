import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def read_data(file_path):
    """
    Read data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    dataframe: The data read from the CSV file.
    """
    # Read the CSV file into a DataFrame
    data = pd.read_csv(file_path)

    return data

def data_find_duplicates(data):
    """
    Find duplicate columns in the data.

    Parameters:
    data (dataframe): The input data to check for duplicates.

    Returns:
    dataframe: A DataFrame containing the duplicate rows.
    """

    # Check for duplicated columns
    duplicated_cols = data.columns.duplicated()
    print("Duplicated columns:", data.columns[duplicated_cols])

    # Check for columns with entirely missing data
    missing_cols = data.isnull().all()
    print("Columns with entirely missing data:", data.columns[missing_cols])




def data_preprocessing(data):
    """
    Preprocess the input data by deleting the completely empty columns, and formatting the date.

    Parameters:
    data (dataframe): The input data to be preprocessed.

    Returns:
    list of float: The preprocessed data.
    """

    # Delete completely empty columns
    data = data.dropna(axis=1, how='all')

    # Format the date column
    data['transactionDateTime_dt'] = pd.to_datetime(data['transactionDateTime'], format='%Y-%m-%dT%H:%M:%S')
    data['currentExpDate_dt'] = pd.to_datetime(data['currentExpDate'], format='%m/%Y')
    data['accountOpenDate_dt'] = pd.to_datetime(data['accountOpenDate'], format='%Y-%m-%d')
    data['dateOfLastAddressChange_dt'] = pd.to_datetime(data['dateOfLastAddressChange'], format='%Y-%m-%d')

    data.drop(columns=['transactionDateTime', 'currentExpDate', 'accountOpenDate', 'dateOfLastAddressChange'], inplace=True)
    data.to_csv('data/processed/preprocessed_data.csv', index=False)
    return data



def train_test_split_data(data, target_column, test_size=0.3, random_state=42):
    """
    Split the data into training and testing sets.

    Parameters:
    data (dataframe): The input data to be split.
    target_column (str): The name of the target column.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.

    Returns:
    tuple: The training and testing sets (X_train, X_test, y_train, y_test).
    """

    ## only keep the numerical data for model training
    numeric_data = data.select_dtypes(include=[np.number])
    data = numeric_data.copy()
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test