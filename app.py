from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return "API is working"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['input']
        
        scaled_data = scaler.transform([data])
        result = model.predict(scaled_data)
        
        return jsonify({'result': result.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)