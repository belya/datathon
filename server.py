from flask import Flask, jsonify, request
from sklearn.externals import joblib
import pandas as pd

app = Flask(__name__)

@app.route('/prediction', methods=['POST'])
def prediction():
  return jsonify({'value': 0.85})

@app.route('/predictions', methods=['POST'])
def predictions():
  return jsonify([{'value': 0.85, 'userNumber': user["userNumber"]} for user in request.json])

if __name__ == '__main__':
  app.run(port=8080)
