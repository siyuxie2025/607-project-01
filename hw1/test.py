from data_preprocessing import read_data, data_find_duplicates, data_preprocessing, train_test_split_data

if __name__ == "__main__":
    # Example usage
    file_path = 'data/raw/CreditCardFraud_updated.csv'  # Replace with your actual file path
    data = read_data(file_path)

    data_find_duplicates(data)

    preprocessed_data = data_preprocessing(data)
    
    X_train, X_test, y_train, y_test = train_test_split_data(preprocessed_data, target_column='isFraud')
    print("Data preprocessing and splitting completed.")