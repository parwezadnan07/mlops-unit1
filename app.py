from flask import Flask, request, jsonify
import joblib
import random

app = Flask(__name__)

# Load both versions
blue_v1 = joblib.load('models/iris_blue.pkl')
green_v2 = joblib.load('models/iris_green.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Iris expects 4 features: sepal length, sepal width, petal length, petal width
    features = data['features']
    
    # A/B Testing Logic: Randomly route traffic
    if random.random() < 0.5:
        prediction = green_v2.predict([features])
        version = "Green (Random Forest)"
    else:
        prediction = blue_v1.predict([features])
        version = "Blue (Logistic Regression)"
        
    return jsonify({
        'prediction': int(prediction[0]),
        'deployed_version': version
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)