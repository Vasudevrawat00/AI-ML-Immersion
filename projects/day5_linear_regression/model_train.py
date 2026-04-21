import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# =========================
# DATASET
# =========================
data = {
    'Hours': [1,2,3,4,5,6,7,8,9,10],
    'Score': [35,40,55,60,68,72,81,88,92,95]
}

df = pd.DataFrame(data)

X = df[['Hours']]
y = df['Score']

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training items: {len(X_train)} | Testing items: {len(X_test)}")

# =========================
# MODEL TRAINING
# =========================
model = LinearRegression()
model.fit(X_train, y_train)

# =========================
# PREDICTION
# =========================
predictions = model.predict(X_test)

print("\nPredictions:", predictions)
print("Actual:", y_test.values)

# Extra prediction (important for checklist)
pred_11 = model.predict([[11]])
print(f"\nPrediction for 11 hours: {pred_11[0]:.2f}")

# =========================
# EVALUATION
# =========================
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"\nMean Squared Error: {mse:.2f}")
print(f"R-Squared Score: {r2:.2f}")

# =========================
# VISUALIZATION
# =========================
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.title("Hours vs Score: AI Prediction Line")
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.show()