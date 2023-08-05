import pytest
import joblib
from starter.ml.model import compute_model_metrics
from starter.ml.data import process_data
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

@pytest.fixture(scope="session")
def data():

    dataset = pd.read_csv('data/census.csv')
    encoder = joblib.load('model/encoder.joblib')
    lb = joblib.load('model/lb.joblib')
    model = joblib.load('model/model.sav')
    train, test = train_test_split(dataset, test_size=0.20)

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

    X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder= encoder, lb=lb
    )
    y_pred = model.predict(X_test)

    return model, train, X_test, y_test, y_pred, cat_features


def test_compute_model_metrics(data):

    model, train, X_test, y_test, y_pred, cat_features= data

    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

    assert isinstance(precision, (np.float64, float))
    assert isinstance(recall, (np.float64, float))
    assert isinstance(fbeta, (np.float64, float))

def test_inference(data):

    model, train, X_test, y_test, y_pred, cat_features= data
    
    preds = model.predict(X_test)

    assert len(preds) > 0

def test_process_data(data):

    model, train, X_test, y_test, y_pred, cat_features = data

    X_train, y_train, _, _ = process_data(
    train, categorical_features=cat_features, label="salary", training=True
    )

    assert len(X_train) > 0
    assert len(y_train) > 0

