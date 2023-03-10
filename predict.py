from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

def predict_single(customer, model):
    customer = pd.DataFrame(customer, index=[0])
    proba = model.predict_proba(customer)[:,1]
    y_pred = model.predict(customer)
    return y_pred, proba


with open('churn_model.bin', 'rb') as f_in:
    model = pickle.load((f_in))

app = Flask('churn_prediction')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    prediction, proba = predict_single(customer, model)

    result = {
        'Churn': bool(prediction),
        'Probability': float(proba)
        }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)