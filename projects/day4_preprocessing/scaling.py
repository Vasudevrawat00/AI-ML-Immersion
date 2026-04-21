from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Sample data: [Age, Salary]
data = np.array([
    [22, 50000],
    [45, 120000],
    [30, 80000]
])

# Initialize scaler
scaler = MinMaxScaler()

# Scale the data between 0 and 1
scaled_data = scaler.fit_transform(data)

print("Original Data:\n", data)
print("\nScaled Data:\n", scaled_data)