**This is the first project of Stats 607 25FA.**

## Project Instructions
The goal of this project is to learn: 

1. organize messy code into structured, maintainable projects. 

2. automate your workflow to make reproducibility effortless


## Data Analysis Description
This was a homework to perform exploratory data analysis and data cleaning on a dataset of credit card transactions. 
The primary goal of this project was to gain insights into the predictors of credit card fraud, which is an important problem in the financial industry.
The dataset simulates credit card transaction info resembling that of a financial institution's customers. 
In addition to the original exploratory data analysis, I train a logistic regression model to predict whether a transaction would be a fraud. 

## Dataset
The raw data is available at [CreditCardFraud_updated](https://www.dropbox.com/scl/fi/amlnrhnefhwkyv6byn517/CreditCardFraud_updated.csv?rlkey=t6hnsl3w5c77xddtkxzjxasa1&st=fy22wrtx&dl=0). 

## Dataset Description

The following variables are included in the dataset:

- `accountNumber`: a unique identifier for the customer account associated with the transaction
- `customerId`: a unique identifier for the customer associated with the transaction
- `creditLimit`: the maximum amount of credit available to the customer on their account
- `availableMoney`: the amount of credit available to the customer at the time of the transaction
- `transactionDateTime`: the date and time of the transaction
- `transactionAmount`: the amount of the transaction
- `merchantName`: the name of the merchant where the transaction took place
- `acqCountry`: the country where the acquiring bank is located
- `merchantCountryCode`: the country where the merchant is located
- `posEntryMode`: the method used by the customer to enter their payment card information during the transaction
- `posConditionCode`: the condition of the point-of-sale terminal at the time of the transaction
- `merchantCategoryCode`: the category of the merchant where the transaction took place
- `currentExpDate`: the expiration date of the customer's payment card
- `accountOpenDate`: the date the customer's account was opened
- `dateOfLastAddressChange`: the date the customer's address was last updated
- `cardCVV`: the three-digit CVV code on the back of the customer's payment card
- `enteredCVV`: the CVV code entered by the customer during the transaction
- `cardLast4Digits`: the last four digits of the customer's payment card
- `transactionType`: the type of transaction
- `echoBuffer`: an internal variable used by the financial institution
- `currentBalance`: the current balance on the customer's account
- `merchantCity`: the city where the merchant is located
- `merchantState`: the state where the merchant is located
- `merchantZip`: the ZIP code where the merchant is located
- `cardPresent`: whether or not the customer's payment card was present at the time of the transaction
- `posOnPremises`: whether or not the transaction took place on the merchant's premises
- `recurringAuthInd`: whether or not the transaction was a recurring payment
- `expirationDateKeyInMatch`: whether or not the expiration date of the payment card was entered correctly during the transaction
- `isFraud`: whether or not the transaction was fraudulent


## Setup Instructions
Follow `SETUP.md` to set up the environment and reqiured packages. 

## Usage Examples
### Example 1: Preprocessing
```{python}
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from data_preprocessing import data_preprocessing, data_save_processed_data

data = pd.read_csv('data/raw/CreditCardFraud_updated.csv')
preprocessed_data = data_preprocessing(data)
data_save_processed_data(preprocessed_data, "data/processed/preprocessed_data")
```

### Example 2: Build Logistic Regression Model and Save Results
```{python}
from data_preprocessing import train_test_split_data
import pandas as pd
import numpy as np

preprocessed_data = pd.read_csv("data/processed/preprocessed_data")
X_train, X_test, y_train, y_test = train_test_split_data(preprocessed_data, target_column = 'isFraud')
run_analysis(X_train, X_test, y_train, y_test)
```
Results are saved in results/report folder. 

### Example 3: Exploratory Data Analysis
```{python}
from EDA import eda_features, eda_cvv_match, eda_transaction_amount, eda_categorical_features
import pandas as pd
import numpy as np

preprocessed_data = pd.read_csv("data/processed/preprocessed_data")
eda_features(preprocessed_data)
eda_cvv_match(preprocessed_data)
eda_transaction_amount(preprocessed_data)
eda_categorical_features(preprocessed_data)
```
Figures are saved in results/figures folder. 