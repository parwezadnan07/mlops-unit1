import pandas as pd

# Loading the Iris dataset from a remote URL
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

print("--- Dataset Statistics ---")
print(df.describe())

print("\n--- Feature Correlation Matrix ---")
print(df.drop('species', axis=1).corr())