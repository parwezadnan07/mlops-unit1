import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# 1. Load the versioned dataset
df = pd.read_csv('data/diabetes.csv')

# 2. Split data
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Save Model to the models folder
os.makedirs('models', exist_ok=True)
model_path = 'models/diabetes_model.pkl'
joblib.dump(model, model_path)

# 5. Output Metrics
predictions = model.predict(X_test)
print(f"Diabetes Model Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
print(f"Model saved to: {model_path}")