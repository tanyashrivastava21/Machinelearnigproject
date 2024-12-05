import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained model and scaler
model_path = 'model_svc.pkl'
scaler_path = 'scaler.pkl'

try:
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    print("Model and Scaler loaded successfully!")
except Exception as e:
    print(f"Error loading model or scaler: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        features = [
            float(request.form['age']),
            int(request.form['gender']),
            int(request.form['chest_pain']),
            float(request.form['resting_bp']),
            float(request.form['cholesterol']),
            int(request.form['fasting_bs']),
            int(request.form['rest_ecg']),
            float(request.form['max_hr']),
            int(request.form['exercise_angina']),
            float(request.form['st_depression']),
            int(request.form['slope']),
            int(request.form['ca']),
            int(request.form['thal'])
        ]

        # Convert to numpy array and scale features
        features_array = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(features_array)

        # Make prediction
        prediction = model.predict(scaled_features)[0]
        prediction_proba = model.predict_proba(scaled_features)[0][1]

        # Prepare the output
        result = "Disease Detected" if prediction == 1 else "No Disease"
        probability_text = f"Probability of Disease: {prediction_proba:.2f}"

        return render_template(
            'index.html',
            prediction_text=f"Prediction: {result}",
            probability_text=probability_text
        )
    except Exception as e:
        return render_template(
            'index.html',
            prediction_text="Error: Could not process input",
            probability_text=f"Details: {str(e)}"
        )

if __name__ == '__main__':
    app.run(debug=True)
