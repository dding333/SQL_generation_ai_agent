import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import numpy as np
import os

# Load the dataset and configure pandas to display larger column widths
data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
pd.set_option('max_colwidth', 200)

# Split data into training and testing sets (80% train, 20% test)
train_set, test_set = train_test_split(data, test_size=0.20, random_state=42)
train_set = train_set.reset_index(drop=True)
test_set = test_set.reset_index(drop=True)

# Define train data subsets based on different user information categories
train_demographics = train_set[['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents']]

train_services = train_set[['customerID', 'PhoneService', 'MultipleLines', 'InternetService',
                            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                            'StreamingTV', 'StreamingMovies']]

train_payments = train_set[['customerID', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                            'MonthlyCharges', 'TotalCharges']]

train_churn = train_set[['customerID', 'Churn']]

# Define test data subsets based on the same user information categories
test_demographics = test_set[['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents']]

test_services = test_set[['customerID', 'PhoneService', 'MultipleLines', 'InternetService',
                          'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                          'StreamingTV', 'StreamingMovies']]

test_payments = test_set[['customerID', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                          'MonthlyCharges', 'TotalCharges']]

test_churn = test_set[['customerID', 'Churn']]

# Seed setting for reproducibility in random operations
np.random.seed(42)

# 1. Randomly drop 5% of rows from the demographics training data
remove_indices = np.random.choice(train_demographics.index, size=int(0.05 * len(train_demographics)), replace=False)
train_demographics = train_demographics.drop(remove_indices)

# 2. Append new customer IDs to the services training data
new_customer_ids = ["NEW" + str(i) for i in range(100)]
new_customers_df = pd.DataFrame({'customerID': new_customer_ids})
train_services = pd.concat([train_services, new_customers_df], ignore_index=True)

# 3. Introduce missing values into random positions within the payments training data
for _ in range(100):
    rand_row = np.random.randint(train_payments.shape[0])
    rand_col = np.random.randint(1, train_payments.shape[1])  # Skip the 'customerID' column
    train_payments.iat[rand_row, rand_col] = np.nan

# 4. Add new customer churn data to the churn training data
additional_churn_ids = ["NEWCHURN" + str(i) for i in range(50)]
new_churn_df = pd.DataFrame({'customerID': additional_churn_ids, 'Churn': ['Yes'] * 25 + ['No'] * 25})
train_churn = pd.concat([train_churn, new_churn_df], ignore_index=True)

# Display information about the test churn data
print(test_churn.info())

# Create a directory if it doesn't already exist
if not os.path.exists('telco_data'):
    os.makedirs('telco_data')

# Uncomment the following lines to save the dataframes to CSV files:
# train_demographics.to_csv('telco_data/train_demographics.csv', index=False)
# train_services.to_csv('telco_data/train_services.csv', index=False)
# train_payments.to_csv('telco_data/train_payments.csv', index=False)
# train_churn.to_csv('telco_data/train_churn.csv', index=False)
#
# test_demographics.to_csv('telco_data/test_demographics.csv', index=False)
# test_services.to_csv('telco_data/test_services.csv', index=False)
# test_payments.to_csv('telco_data/test_payments.csv', index=False)
# test_churn.to_csv('telco_data/test_churn.csv', index=False)

if __name__ == '__main__':
    print("This script processes the Telco-Customer CSV file into separate tables and stores them in the 'telco_data' directory.")


