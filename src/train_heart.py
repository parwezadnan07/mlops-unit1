import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import joblib
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

print("Generating synthetic Heart Disease dataset...")
# Create a synthetic dataset that looks like the heart disease data
X_raw, y_raw = make_classification(n_samples=1000, n_features=13, n_classes=2, random_state=42)

# Convert to DataFrame to maintain the expected structure for app.py
columns = [f'feature_{i}' for i in range(13)]
df = pd.DataFrame(X_raw, columns=columns)
X = df
y = y_raw

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training the model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
model_path = 'models/heart_model.pkl'
joblib.dump(model, model_path)
print(f"Success! Model saved to: {model_path}")