import requests

url = "http://127.0.0.1:9696/predict"
client = {"job": "retired", "duration": 445, "poutcome": "success"}

response = requests.post(url, json=client)

print(response.json())