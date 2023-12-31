# Script to train machine learning model.
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_performence_on_data_slices
from starter.ml.model import compute_model_metrics
# Add the necessary imports for the starter code.
import pandas as pd
import joblib
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
model = train_model(X_train, y_train)
filename = 'model/model.sav'
joblib.dump(model, filename)
joblib.dump(encoder, 'model/encoder.joblib')
joblib.dump(lb, 'model/lb.joblib')

X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features,
        label="salary", training=False, encoder=encoder, lb=lb
)

y_pred = model.predict(X_test)

precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

print('precision, recall, fbeta', precision, recall, fbeta)

# Compute model metrics on slices of the data
slice = ['education']
compute_performence_on_data_slices(
    model, test, slice, cat_features, encoder, lb
)
