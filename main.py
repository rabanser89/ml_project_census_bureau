# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel

from starter.ml.model import inference
from starter.ml.data import process_data
import joblib
import pandas as pd

encoder = joblib.load('model/encoder.joblib')
lb = joblib.load('model/lb.joblib')
model = joblib.load('model/model.sav')
data0 = pd.read_csv('data/census.csv')
cols = list(data0.columns)

print(data0.shape)

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
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str
   

app = FastAPI()

@app.get("/")
async def say_hello():
    return {"greeting": "Hello World"}

@app.post("/inference")
async def model_inference(data: ModelInput):

    
    df = pd.DataFrame.from_dict({0:list(dict(data).values())}, orient='index',
                       columns=cols[:-1])

    X, _, _, _ = process_data(
        df, categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=None
    )
    preds = inference(model, X)
    r = lb.inverse_transform(preds)
    result = {'pred': r[0]}
    return result