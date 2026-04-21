import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# =========================
# GENERATE DATA
# =========================
np.random.seed(42)
X = 6 * np.random.rand(100, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1)

# =========================
# POLYNOMIAL MODEL
# =========================
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# =========================
# DECISION TREES (DIFFERENT DEPTHS)
# =========================
tree2 = DecisionTreeRegressor(max_depth=2)
tree5 = DecisionTreeRegressor(max_depth=5)
tree20 = DecisionTreeRegressor(max_depth=20)

tree2.fit(X, y)
tree5.fit(X, y)
tree20.fit(X, y)

# =========================
# PREDICTION LINE
# =========================
X_new = np.linspace(-3, 3, 100).reshape(100, 1)

# Polynomial prediction
X_new_poly = poly_features.transform(X_new)
y_poly = poly_model.predict(X_new_poly)

# Tree predictions
y_tree2 = tree2.predict(X_new)
y_tree5 = tree5.predict(X_new)
y_tree20 = tree20.predict(X_new)

# =========================
# R2 SCORES
# =========================
poly_r2 = r2_score(y, poly_model.predict(X_poly))
tree2_r2 = r2_score(y, tree2.predict(X))
tree5_r2 = r2_score(y, tree5.predict(X))
tree20_r2 = r2_score(y, tree20.predict(X))

print("Polynomial R2:", round(poly_r2, 4))
print("Tree depth=2 R2:", round(tree2_r2, 4))
print("Tree depth=5 R2:", round(tree5_r2, 4))
print("Tree depth=20 R2:", round(tree20_r2, 4))

# =========================
# PLOT
# =========================
plt.scatter(X, y, label="Data")
plt.plot(X_new, y_poly, color='red', label="Polynomial")
plt.plot(X_new, y_tree2, color='green', label="Tree depth=2")
plt.plot(X_new, y_tree5, color='orange', label="Tree depth=5")
plt.plot(X_new, y_tree20, color='purple', label="Tree depth=20")

plt.legend()
plt.title("Polynomial vs Decision Tree")
plt.show()