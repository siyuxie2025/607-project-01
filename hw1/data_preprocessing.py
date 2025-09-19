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
    data = pd.read_csv(file_path)
    return data

def data_preprocessing(data):
    """
    Preprocess the input data by deleting the completely empty columns, filling missing values, and formatting date columns.

    Parameters:
    data (dataframe): The input data to be preprocessed.

    Returns:
    dataframe: The preprocessed data.
    """
    # Step 1: Handle missing values
    data = data_preprocessing_na(data)

    # Step 2: Locate time columns
    time_columns = data_locate_time_columns(data)

    # Step 3: Format date columns
    data = data_preprocessing_dt(data, time_columns)

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

def data_preprocessing_na(data):
    """
    Preprocess the input data by deleting the completely empty columns, and filling missing values.

    Parameters:
    data (dataframe): The input data to be preprocessed.

    Returns:
    list of float: The preprocessed data.
    """

    # Delete completely empty columns
    data = data.dropna(axis=1, how='all')

    return data



def data_preprocessing_dt(data, time_columns):
    """
    Preprocess the input data by deleting the completely empty columns, and formatting the date.

    Parameters:
    data (dataframe): The input data to be preprocessed.

    Returns:
    list of float: The preprocessed data.
    """

    # Format the date column
    for col in time_columns:
        if col in data.columns:
            print(f"Formatting column: {col}")
            col_format = col + '_dt'
            if col == 'transactionDateTime':
                data.loc[:, col_format] = pd.to_datetime(data[col], format='%Y-%m-%dT%H:%M:%S')
            elif col == 'currentExpDate':
                data.loc[:, col_format] = pd.to_datetime(data[col], format='%m/%Y')
            elif col == 'accountOpenDate':
                data.loc[:, col_format] = pd.to_datetime(data[col], format='%Y-%m-%d')
            elif col == 'dateOfLastAddressChange':
                data.loc[:, col_format] = pd.to_datetime(data[col], format='%Y-%m-%d')
            else:
                print(f"Unknown date format for column: {col}")
        else:
            print(f"Column {col} not found in data.")

    # Drop the original date columns
    data.drop(columns=time_columns, inplace=True)
    
    return data

def data_save_processed_data(data, file_path):
    """
    Save the preprocessed data to a CSV file.

    Parameters:
    data (dataframe): The preprocessed data to be saved.
    file_path (str): The path to save the CSV file.

    Returns:
    None
    """
    data.to_csv(file_path, index=False)
    print(f"Preprocessed data saved to {file_path}")

# data.to_csv('data/processed/preprocessed_data.csv', index=False)


def data_locate_time_columns(data):
    """
    Locate the time columns in the data.

    Parameters:
    data (dataframe): The input data to locate time columns.

    Returns:
    list of str: The list of time column names.
    """
    time_columns = [col for col in data.columns if 'Date' in col or 'Time' in col or 'date' in col]
    return time_columns


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

    # only keep the numerical data for model training
    numeric_data = data.select_dtypes(include=[np.number])
    y = data[target_column].astype(int)
    data = numeric_data.copy()
    X = data.to_numpy()
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test