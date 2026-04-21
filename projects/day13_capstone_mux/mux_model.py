import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# STEP 1: CREATE DATA
# =========================
np.random.seed(42)

data = pd.DataFrame({
    "age_diff": np.random.randint(0, 15, 100),
    "common_interests": np.random.randint(0, 10, 100),
    "communication_score": np.random.randint(1, 10, 100),
    "activity_match": np.random.randint(1, 10, 100)
})

# Create target (compatibility logic)
data["compatible"] = (
    (data["common_interests"] > 5) &
    (data["communication_score"] > 5)
).astype(int)

print("\nSample Data:\n", data.head())

# =========================
# STEP 2: SPLIT FEATURES & TARGET
# =========================
X = data.drop("compatible", axis=1)
y = data["compatible"]

# =========================
# STEP 3: POLYNOMIAL FEATURES
# =========================
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# =========================
# STEP 4: TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)

# =========================
# STEP 5: SCALING
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# STEP 6: RANDOM FOREST MODEL
# =========================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# =========================
# STEP 7: PREDICTION
# =========================
y_pred = model.predict(X_test)

# =========================
# STEP 8: ACCURACY
# =========================
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

# =========================
# STEP 9: CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# =========================
# STEP 10: FEATURE IMPORTANCE
# =========================
importances = model.feature_importances_
feature_names = poly.get_feature_names_out(X.columns)

feat_importance = pd.Series(importances, index=feature_names)

print("\nTop 5 Important Features:")
print(feat_importance.nlargest(5))