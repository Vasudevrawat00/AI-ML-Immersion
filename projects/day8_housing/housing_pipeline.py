from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# =========================
# SECTION 1: LOAD DATA
# =========================
data = fetch_california_housing()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target

print("Dataset Overview:")
print(df.head())

print("\nBasic Statistics:")
print(df.describe())

# =========================
# SECTION 2: PREPROCESSING
# =========================
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# SECTION 3: TRAIN + EVALUATE
# =========================
model = LinearRegression()
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"\nMean Absolute Error: ${mae * 100000:.2f}")
print(f"R-Squared Score: {r2:.2f}")

# =========================
# SECTION 4: RESIDUAL PLOT
# =========================
residuals = y_test - predictions

plt.scatter(y_test, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Actual Price")
plt.ylabel("Error (Residuals)")
plt.title("Residual Plot")
plt.show()