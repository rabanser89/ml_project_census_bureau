from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import logging
from starter.ml.data import process_data
# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a random forest classifier and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta



def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds

logging.basicConfig(
    filename='slice_output.txt',
    level=logging.INFO,
    filemode='w',
    format='%(message)s')

def compute_performence_on_data_slices(model, test_data, slice, cat_features, encoder, lb):
    """
    compute model performance on data slices 
    Inputs
    ------
    model : ???
        Trained machine learning model.
    test_data : ???
        Data used for testing.
    slice : lst
        list of slices where performance is calculated.
    cat_features: lst
        list of categorical features
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer
    Returns
    -------
    None
    """

    df = test_data[slice]
    for cat in df.columns:
        values = list(set(df[cat]))
        for val in values:
            data_slice = test_data[test_data[cat]==val]
            X, y, _, _ = process_data(
                    data_slice, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
                )
            preds = inference(model, X)
            precision, recall, fbeta = compute_model_metrics(y, preds)
            logging.info(f"Model performance for {cat} with value {val}: precision = {precision}, recall = {recall}, fbeta = {fbeta}")
            
            
    
