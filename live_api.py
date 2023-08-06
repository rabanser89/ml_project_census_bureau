import requests
import json

data = 0
response = requests.post('/url/to/query/', data=json.dumps(data))

print(response.status_code)
print(response.json())