import numoy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import read_data, data_preprocessing, data_find_duplicates, train_test_split_data

def eda_features(data):
    """
    Perform exploratory data analysis on numerical features, whether CVV matches.

    Parameters:
    data (dataframe): The input data for EDA.

    Returns:
    None
    """
    # List of numerical columns to analyze
    numeric_cols = ["creditLimit", "currentBalance","availableMoney", "transactionAmount"]
    
    # Create a figure to hold all subplots
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 20))

    # Loop through each numeric column
    for i, col in enumerate(numeric_cols):
        sns.boxplot(data=data[col], ax=axes[i, 0])
        sns.histplot(data=data[col], bins=100, ax=axes[i, 1], color='skyblue')
    plt.tight_layout()
    plt.savefig('results/figures/boxplot_histogram_combined.png')
    plt.show()

def eda_cvv_match(data):
    """
    Perform exploratory data analysis on CVV matching features.

    Parameters:
    data (dataframe): The input data for EDA.

    Returns:
    None
    """
    # Create a new column to indicate if CVV matches
    # for all data
    data['is_cvv_match'] = (data['CVVCode'] == data['CVVCodeMatch']).astype(int)
    sns.countplot(data=data, x='is_cvv_match', hue='is_cvv_match', palette='Set2')
    plt.title('Relationship between isFraud and is_cvv_match')
    plt.xlabel('isFraud')
    plt.ylabel('Count')
    plt.savefig('results/figures/is_cvv_match_countplot.png')
    plt.show()

    # for fraud data
    sns.countplot(data=data[data.isFraud==1], x='is_cvv_match', hue='is_cvv_match')
    plt.title('Barplot of CVV mismatching (for fraud data)')
    plt.xlabel('is_cvv_match')
    plt.ylabel('Count')
    plt.savefig('results/figures/is_cvv_match_fraud_countplot.png')
    plt.show()

def eda_transaction_amount(data):
    """
    Perform exploratory data analysis on transaction amounts.

    Parameters:
    data (dataframe): The input data for EDA.

    Returns:
    None
    """
    # Plot the distribution of transaction amounts
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='transactionAmount', bins=100, color='skyblue')
    plt.title('Distribution of Transaction Amounts')
    plt.xlabel('Transaction Amount')
    plt.ylabel('Frequency')
    plt.savefig('results/figures/transaction_amount_distribution.png')
    plt.show()

    # Plot the distribution of transaction amounts with a zoomed-in view
    sns.histplot(data=data[data.transactionAmount < 1500], x='transactionAmount', bins=30)
    plt.xlabel('Transaction Amount')
    plt.ylabel('Frequency')
    plt.title('Distribution of Transaction Amount (Zoomed In)')
    plt.savefig('results/figures/transaction_amount_distribution_zoomed.png')
    plt.show()

    # Plot the log-transformed transaction amounts
    sns.histplot(np.log(data.transactionAmount+1), bins=30)
    plt.xlabel('log(Transaction Amount)')
    plt.ylabel('Frequency')
    plt.title('Distribution of log(Transaction Amount)')
    plt.savefig('results/figures/log_transaction_amount_distribution.png')
    plt.show()

def eda_categorical_features(data):
    """
    Perform exploratory data analysis on categorical features.

    Parameters:
    data (dataframe): The input data for EDA.

    Returns:
    None
    """
    categorical_cols = ['merchantCategoryCode', 'posEntryMode', 'transactionType', 'posConditionCode', 'merchantCountryCode']

    # Create subplots
    fig, axes = plt.subplots(len(categorical_cols), 1, figsize=(8, 6*len(categorical_cols)))

    # Iterate over each categorical predictor
    for i, predictor in enumerate(categorical_cols):
        # Calculate fraud rate for each category
        fraud_rate = data.groupby(predictor)['isFraud'].mean().sort_values(ascending=False)

        # Create bar plot
        sns.barplot(x=fraud_rate.index, y=fraud_rate.values, ax=axes[i])
        axes[i].set_title(f'Fraud Rate by {predictor}')
        axes[i].set_ylabel('Fraud Rate')
        axes[i].set_xlabel(predictor)
        axes[i].tick_params(axis='x', rotation=45)

    # Adjust layout
    plt.tight_layout()
    plt.savefig('results/figures/fraud_rate_by_categorical_features.png')
    plt.show()


    # Create a figure to hold all subplots
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 20))

    # Loop through each categorical column
    for i, col in enumerate(categorical_cols):
        sns.countplot(data=data, x=col, hue='isFraud', ax=axes[i//2, i%2], palette='Set2')
        axes[i//2, i%2].set_title(f'Countplot of {col} by isFraud')
        axes[i//2, i%2].set_xlabel(col)
        axes[i//2, i%2].set_ylabel('Count')
        axes[i//2, i%2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('results/figures/categorical_features_countplots.png')
    plt.show()

    ## full data
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Iterate over each variable
    for i, col in enumerate(['transactionAmount', 'creditLimit', 'availableMoney', 'currentBalance']):
        # Plot the conditional probability density plot for fraud cases (blue)
        sns.kdeplot(data=df2[df2.isFraud==True][col], fill=True, ax=axes[i // 2, i % 2], label="is_fraud")
        # Plot the conditional probability density plot for non-fraud cases (orange)
        sns.kdeplot(data=df2[df2.isFraud==False][col], fill=True, ax=axes[i // 2, i % 2], label="not_fraud")
        axes[i // 2, i % 2].set_title(f'Conditional Probability Density of {col} by isFraud')
        axes[i // 2, i % 2].set_xlabel(col)
        axes[i // 2, i % 2].set_ylabel('Density')
        axes[i // 2, i % 2].legend()

    # Adjust layout
    plt.tight_layout()
    plt.savefig('results/figures/conditional_probability_density_plots.png')
    plt.show()


# Example usage:
# if __name__ == "__main__":
#     # Load and preprocess the data
#     file_path = 'CreditCardFraud_updated.csv'  # Replace with your actual file path
#     data = read_data(file_path)
#     data_find_duplicates(data)
#     preprocessed_data = data_preprocessing(data)
#     eda_features(preprocessed_data)
#     eda_cvv_match(preprocessed_data)
#     eda_transaction_amount(preprocessed_data)
#     eda_categorical_features(preprocessed_data)
