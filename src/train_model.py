import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# 1. Load dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# 2. Split data into Features (X) and Target (y)
X = df.drop('species', axis=1)
y = df['species']

# 3. Split into Training and Testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train a simple ML model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 5. Print Evaluation Metrics
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 6. Save the trained model
joblib.dump(model, 'iris_model.pkl')
print("Model saved as iris_model.pkl")