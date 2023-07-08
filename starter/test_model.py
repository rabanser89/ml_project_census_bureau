import joblib
from ml.model import compute_model_metrics, inference
from ml.data import process_data
# Load model
model = joblib.load('../model/model.sav')

def test_compute_model_metrics(y, preds):

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert type(precision) == float
    assert type(recall) == float
    assert type(fbeta) == float

def test_inference(model, X):
    
    preds = model.predict(X)

    assert len(preds) > 0

def test_process_data(train, cat_features):

    X_train, y_train, _, _ = process_data(
    train, categorical_features=cat_features, label="salary", training=True
    )

    assert len(X_train) > 0
    assert len(y_train) > 0

