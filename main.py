# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel

from starter.ml.model import inference
from starter.ml.data import process_data
from starter.train_model import cat_features
import joblib
import pandas as pd
import numpy as np

encoder = joblib.load('model/encoder.joblib')
model = joblib.load('model/model.sav')
cols = list(pd.read_csv('data/census.csv').columns)

#to do adapt
class ModelInput(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex:str
    capital_gain: int
    capital_loss:int
    hours_per_week:int
    native_country: str


app = FastAPI()

@app.get("/")
async def say_hello():
    return {"greeting": "Hello World"}

@app.post("inference")
async def model_inference(data: ModelInput):

    df = pd.DataFrame.from_dict({0:list(data.values())[:-1]}, orient='index',
                       columns=cols)
    X, _, _, _ = process_data(
        df, categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=None
    )
    preds = inference(model, X)
    return preds

