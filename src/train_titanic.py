import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load Titanic Data
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Simple Preprocessing
df = df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].dropna()
X = df.drop('Survived', axis=1)
y = df['Survived']

model = RandomForestClassifier().fit(X, y)
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/titanic_model.pkl')
print("Titanic model trained and saved!")