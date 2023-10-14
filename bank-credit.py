import json
import pickle
import pandas as pd

with open("dv.bin", "rb") as infile:
    dv = pickle.load(infile)

with open("model1.bin", "rb") as infile:
    model = pickle.load(infile)

client_data = dv.transform(
    {"job": "retired", "duration": 445, "poutcome": "success"}
)

pred = model.predict_proba(client_data)[0], model.predict(client_data)[0]
print(pred)