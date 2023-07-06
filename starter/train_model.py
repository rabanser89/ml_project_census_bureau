# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.data import process_data 
# Add the necessary imports for the starter code.
import pandas as pd
# Add code to load in the data.
data = pd.read_csv('data/census.csv')

# Split training and test data
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process the test data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Train and save a model.
