from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# =========================
# LOAD DATA
# =========================
data = fetch_california_housing()

X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# =========================
# SCALE DATA
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# GRID SEARCH
# =========================
param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0, 500.0]}

grid_search = GridSearchCV(Ridge(), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train_scaled, y_train)

print("Best Alpha Found:", grid_search.best_params_)
print(f"Best Cross-Validation R2: {grid_search.best_score_:.4f}")

# =========================
# COMPARISON (IMPORTANT)
# =========================

# Default model
default_model = Ridge()
default_model.fit(X_train_scaled, y_train)
default_pred = default_model.predict(X_test_scaled)
default_r2 = r2_score(y_test, default_pred)

# Tuned model
best_model = grid_search.best_estimator_
best_pred = best_model.predict(X_test_scaled)
best_r2 = r2_score(y_test, best_pred)

print(f"\nDefault Ridge R2: {default_r2:.4f}")
print(f"Tuned Ridge R2: {best_r2:.4f}")