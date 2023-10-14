import pickle

from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

with open("dv.bin", "rb") as infile:
    dv = pickle.load(infile)

with open("model1.bin", "rb") as infile:
    model = pickle.load(infile)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    X = dv.transform([data])
    prediction = model.predict_proba(X)[0]

    result = {
        'prediction': prediction
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
