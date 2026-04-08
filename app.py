import logging
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# 1. Setup Logging Configuration
logging.basicConfig(
    filename='logs/model_monitor.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

model = joblib.load('models/titanic_model.pkl')

# Simple counter for Monitoring/Alerting
death_prediction_count = 0

@app.route('/predict', methods=['POST'])
def predict():
    global death_prediction_count
    data = request.get_json()
    features = data['features']
    
    # Predict
    prediction = model.predict([features])[0]
    result = "Survived" if prediction == 1 else "Did Not Survive"
    
    # 2. Monitoring Logic
    logging.info(f"Input: {features} | Prediction: {result}")
    
    # 3. Simple Alerting Logic
    if prediction == 0:
        death_prediction_count += 1
    else:
        death_prediction_count = 0 # Reset if someone survives

    if death_prediction_count >= 5:
        logging.warning("ALERT: High mortality rate detected in last 5 requests! Check for Data Drift.")
        alert_status = "TRIGGERED"
    else:
        alert_status = "NORMAL"

    return jsonify({
        'prediction': int(prediction),
        'status': result,
        'system_alert': alert_status
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)