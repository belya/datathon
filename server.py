from flask import Flask, jsonify, request
from sklearn.externals import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/prediction', methods=['POST'])
def prediction():
  json_ = request.json
  keys = json_.keys()
  df = pd.DataFrame(data=[json_.values()], columns=keys)
  return jsonify({'value': model.predict_proba(df)[0][1]})

@app.route('/predictions', methods=['POST'])
def predictions():
  json_ = request.json
  keys = json_[0].keys()
  df = pd.DataFrame(data=[user.values() for user in json_], columns=keys)
  return jsonify([{'value': proba[1]} for proba in model.predict_proba(df)])

if __name__ == '__main__':
  model = joblib.load('./model.pkl')
  app.run(port=8080)
