from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

iris = load_iris()
os.makedirs('models', exist_ok=True)

# Blue Model: Simple Logistic Regression
model_blue = LogisticRegression(max_iter=200).fit(iris.data, iris.target)
joblib.dump(model_blue, 'models/iris_blue.pkl')

# Green Model: Random Forest (The 'Improvement')
model_green = RandomForestClassifier().fit(iris.data, iris.target)
joblib.dump(model_green, 'models/iris_green.pkl')

print("Experiment 8: Both Blue and Green models have been generated.")