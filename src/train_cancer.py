import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import json

# 1. Load Data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train
model = RandomForestClassifier(n_estimators=50)
model.fit(X_train, y_train)

# 3. Evaluate
acc = model.score(X_test, y_test)

# 4. Save Metrics for CI/CD to read
with open("metrics.json", "w") as f:
    json.dump({"accuracy": acc}, f)

# 5. Save Model
joblib.dump(model, "models/cancer_model.pkl")