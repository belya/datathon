from flask import Flask, jsonify, request
from sklearn.externals import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/loyalty', methods=['POST'])
def loyalty():
  json_ = request.json
  keys = json_.keys()
  df = pd.DataFrame(data=[json_.values()], columns=keys)
  return jsonify({'value': loyalty_model.predict_proba(df)[0][1]})

@app.route('/loyalties', methods=['POST'])
def loyalties():
  json_ = request.json
  keys = json_[0].keys()
  df = pd.DataFrame(data=[user.values() for user in json_], columns=keys)
  return jsonify([{'value': proba[1]} for proba in loyalty_model.predict_proba(df)])

@app.route('/balance', methods=['POST'])
def balance():
  json_ = request.json
  keys = json_.keys()
  df = pd.DataFrame(data=[json_.values()], columns=keys)
  return jsonify({'value': balance_model.predict(df)[0]})

@app.route('/balances', methods=['POST'])
def balances():
  json_ = request.json
  keys = json_[0].keys()
  df = pd.DataFrame(data=[user.values() for user in json_], columns=keys)
  return jsonify([{'value': proba} for proba in balance_model.predict(df)])

if __name__ == '__main__':
  loyalty_model = joblib.load('./loyalty_model.pkl')
  balance_model = joblib.load('./balance_model.pkl')
  app.run(port=8080)
