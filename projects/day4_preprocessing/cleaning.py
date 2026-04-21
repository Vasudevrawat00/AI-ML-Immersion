import pandas as pd #pd is shortcut to make typing easier 
#pandas is a python library used to work with data
#it helps you read, clean, and analyze datasets
df = pd.read_csv("datasets/data.csv")#This line reads the data.csv file.
#read_csv() loads the data into a table called a DataFrame.
#df stands for DataFrame.
#../../ means:
#Go up one folder from day 4 to projects.
#Go up another folder from projects to AI-ML.
#Enter the datasets folder and open data.csv
print("orignal Data:\n",df)
print("\nMissing values:\n",df.isnull().sum())
df['Age']=df['Age'].fillna(df['Age'].mean())#fillna used to replace missing or null values
#nan= not a  number is unidentified or empty data
df['Score']=df['Score'].fillna(0)
print("\nCleaned data:\n",df)
df.to_csv("datasets/cleaned_data.csv",index=False)
print("\nCleaned dataset saved successfully!")