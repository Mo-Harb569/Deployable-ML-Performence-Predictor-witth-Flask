from flask import Flask, request, render_template
import pickle
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, 'model_files', 'performence_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'model_files', 'scaler.pkl')

try:
    regressor = pickle.load(open(MODEL_PATH, 'rb'))
    scaler = pickle.load(open(SCALER_PATH, 'rb'))
    print("✅ Model and files loaded successfully!")
except FileNotFoundError:
    print("❌ Error: Model files not found. Ensure 'model_files' folder is next to the app.")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    hours = float(request.form['hours'])
    prev_scores = float(request.form['previous_scores'])
    sleep = float(request.form['sleep_hours'])
    papers = float(request.form['sample_papers'])
    activities = int(request.form['activities'])

    features = np.array([[hours, prev_scores, sleep, papers, activities]])

    scaled_features = scaler.transform(features)
    prediction = regressor.predict(scaled_features)
    
    output = round(prediction[0][0], 2)

    return render_template('index.html', prediction_text=f'Predicted Performance Index: {output} / 100')

if __name__ == '__main__':
    app.run(port=5000, debug=True)