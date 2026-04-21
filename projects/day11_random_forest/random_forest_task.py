from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import time

# =========================
# LOAD DATA
# =========================
data = fetch_california_housing()

X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# =========================
# MAIN MODEL
# =========================
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

rf_model.fit(X_train, y_train)

predictions = rf_model.predict(X_test)

print(f"Random Forest R2 Score: {r2_score(y_test, predictions):.4f}")

# =========================
# FEATURE IMPORTANCE
# =========================
importances = rf_model.feature_importances_
feature_names = data.feature_names

feat_importances = pd.Series(importances, index=feature_names)

feat_importances.nlargest(5).plot(kind='barh')
plt.title("Top 5 Most Important Features")
plt.show()

# =========================
# EXPERIMENT: TREE COUNT TEST
# =========================
tree_counts = [10, 50, 200]

print("\nTree Count Experiment:")

for n in tree_counts:
    start = time.time()

    model = RandomForestRegressor(n_estimators=n, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)

    end = time.time()

    print(f"Trees: {n} | R2: {r2:.4f} | Time: {end - start:.2f} sec")