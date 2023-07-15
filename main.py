# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel

from starter.ml.model import inference
from starter.ml.data import process_data
import joblib
import pandas as pd
import numpy as np
import json
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

# data = {
#             "age": 39,
#             "workclass": "Private",
#             "fnlgt": 77516,
#             "education": "Masters",
#             "education_num": 14,
#             "marital_status": "Divorced",
#             "occupation": "Prof-specialty",
#             "relationship": "Not-in-family",
#             "race": "White",
#             "sex": "Male",
#             "capital_gain": 0,
#             "capital_loss": 0,
#             "hours_per_week": 40,
#             "native_country": "United-States"
#         }


# newcol = {cols[:-1][i]:list(data.keys())[i] for i in range(len(data))}
# print(newcol)        
# X0 = data0.rename(columns = newcol)
# X0 = X0[data.keys()]
# print('  X  ', X0.iloc[0,:])
# print('  X2  ', dict(X0.iloc[0,:]))

# D=[]
# for i in range(len(X0)):
#     d = dict(X0.iloc[i,:])
#     df = pd.DataFrame.from_dict({0:list(d.values())}, orient='index',
#                         columns=cols[:-1])

#     X, _, _, _ = process_data(
#         df, categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=None
#     )
#     preds = inference(model,X)
#     print(preds)
#     if preds[0] == 1:
#         print(preds)
#         print(d)
#         y=lb.inverse_transform(preds)
#         print(y)
#         D.append(d)

# with open('data.json', 'w') as f:
#     json.dump(D, f)       

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
    preds = inference(model,X)
    r = lb.inverse_transform(preds)
    result = {'pred': r[0]}
    return result

