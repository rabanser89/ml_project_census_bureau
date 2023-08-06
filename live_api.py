import requests
import json

data = {
            "age": 39,
            "workclass": "Private",
            "fnlgt": 77516,
            "education": "Masters",
            "education_num": 14,
            "marital_status": "Divorced",
            "occupation": "Prof-specialty",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States"
        }

response = requests.get('https://census-bureau.onrender.com')
print('Test live GET:')
print(response.status_code)
print(response.json())

response = requests.post('https://census-bureau.onrender.com/inference', data=json.dumps(data))
print('Test live POST:')
print(response.status_code)
print(response.json())