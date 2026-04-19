import pandas as pd

df = pd.read_csv("practice/data.csv")

print("Table Preview:\n", df.head())

print("\nStatistical Insights:\n", df.describe())

print(f"\nAverage Score: {df['Score'].mean()}")