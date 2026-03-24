import os
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# This finds the exact folder where app.py is sitting
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model.pkl')
scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')

# Now load using the full path
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    print(f"Error: Could not find files at {model_path}")