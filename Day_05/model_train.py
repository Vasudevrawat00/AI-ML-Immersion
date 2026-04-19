import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Dataset
data = {
    'Hours': [1,2,3,4,5,6,7,8,9,10],
    'Score': [35,40,55,60,68,72,81,88,92,95]
}

df = pd.DataFrame(data)

# Input (X) and Output (y)
X = df[['Hours']]
y = df['Score']

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Predict for new value
print("Prediction for 11 hours:", model.predict([[11]])[0])

# Visualization
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.title("Hours vs Score - Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.show()
