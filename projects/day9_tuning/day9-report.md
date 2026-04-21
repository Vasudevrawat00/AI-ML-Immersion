# Day 9 Report – AI/ML Developer Track

**Name:** Vasudev Rawat  
  

## Technical Summary
Today, I learned about hyperparameter tuning using GridSearchCV. I applied Ridge Regression on the California Housing dataset and optimized the alpha parameter using cross-validation.

GridSearchCV helped automate the process of finding the best parameter by testing multiple values and selecting the one with the highest performance.

## Model Performance
- Best Alpha: _______
- Best Cross-Validation R² Score: _______
- Default Ridge R²: _______
- Tuned Ridge R²: _______

## Bug Log
I faced minor issues while running GridSearchCV and handling scaled data. I resolved them by properly applying StandardScaler and ensuring correct data splitting.

## Conceptual Reflection
Using a wider range of values first helps us explore the parameter space more effectively. It allows us to quickly identify the general region where the optimal value lies.

If we start with small increments like [1.1, 1.2, 1.3], we might miss better values far away from this range. A broad search helps locate the best zone, and then we can fine-tune within that range later for better accuracy.

## Tools Used
- Python  
- Scikit-learn  
- NumPy  
- Pandas  