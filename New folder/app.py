from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load your trained model with error handling
model_path = 'model.pkl'
scaler_path = 'scaler.pkl'
label_encoder_path = 'label_encoder.pkl'

# Load the model
if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading the model: {e}")
        model = None  # Handle the failure case
else:
    print(f"Model file '{model_path}' does not exist.")
    model = None  # Handle the failure case

# Load the scaler
if os.path.exists(scaler_path):
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("Scaler loaded successfully.")
    except Exception as e:
        print(f"Error loading the scaler: {e}")
        scaler = None  # Handle the failure case
else:
    print(f"Scaler file '{scaler_path}' does not exist.")
    scaler = None  # Handle the failure case

# Load the label encoder
if os.path.exists(label_encoder_path):
    try:
        with open(label_encoder_path, 'rb') as f:
            le = pickle.load(f)
        print("Label encoder loaded successfully.")
    except Exception as e:
        print(f"Error loading the label encoder: {e}")
        le = None  # Handle the failure case
else:
    print(f"Label encoder file '{label_encoder_path}' does not exist.")
    le = None  # Handle the failure case

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Check if the required data is provided
    if 'age' in data and 'gender' in data and 'vaccine_type' in data:
        age = data['age']
        gender = data['gender']
        vaccine_type = data['vaccine_type']
        
        # Preprocess input for prediction
        input_data = pd.DataFrame([[age, gender, vaccine_type]], columns=['AGE', 'GENDER', 'VACCINE_TYPE'])
        
        # Make prediction
        try:
            # Ensure the scaler and model are available
            if scaler is None or model is None:
                raise ValueError("Model or scaler is not loaded properly.")

            input_data_scaled = scaler.transform(input_data)  # Scale the input data
            prediction = model.predict(input_data_scaled)  # Make prediction
            predicted_label = le.inverse_transform(prediction)  # Decode the prediction
            
            return jsonify({'adverse_reaction': predicted_label[0]})  # Return the prediction
        except Exception as e:
            app.logger.error(f"Prediction error: {e}")  # Log the error
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid input data'}), 400

if __name__ == '__main__':
    app.run(debug=True)
