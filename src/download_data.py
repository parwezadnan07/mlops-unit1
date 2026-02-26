import pandas as pd

# Fetch the dataset used in Exercise 2
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# Save it to the new data folder
df.to_csv('data/iris.csv', index=False)
print("Dataset saved to data/iris.csv")
