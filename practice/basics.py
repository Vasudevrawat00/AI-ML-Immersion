# --- 1. DATA CONTAINERS ---
name = "Developer_Alpha"
age = 24
marks = [85, 92, 78, 95, 88]

# --- 2. THE PROCESSING LOOP ---
for i in marks:
    print(f"Validating Data Point: {i}")

# --- 3. ANALYTICAL FUNCTIONS ---
def analyze_numbers(numbers):
    """Calculates key statistics used in Data Science."""
    min_val = min(numbers)
    max_val = max(numbers)
    avg_val = sum(numbers) / len(numbers)
    return min_val, max_val, avg_val

# Running the Analysis
results = analyze_numbers(marks)

print("\n--- Statistics Report ---")
print(f"Low: {results[0]} | High: {results[1]} | Average: {results[2]}")