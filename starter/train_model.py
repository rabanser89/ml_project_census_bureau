# Script to train machine learning model.
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_performence_on_data_slices
# Add the necessary imports for the starter code.
import pandas as pd
import joblib
# Add code to load in the data.
data = pd.read_csv('../data/census.csv')

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
model = train_model(X_train, y_train)
filename = '../model/model.sav'
joblib.dump(model, filename)

joblib.dump(encoder, '../model/encoder.joblib')


#compute model metrics on slices of the data
slice = ['education'] #slices where performence is computed 
compute_performence_on_data_slices(model, test, slice, cat_features, encoder, lb)
