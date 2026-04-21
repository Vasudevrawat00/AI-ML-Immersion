# Day 10 Report – AI/ML Developer Track

**Name:** Vasudev Rawat  

## Technical Summary
Today, I explored non-linear relationships using Polynomial Regression and Decision Tree Regressor. I generated synthetic curved data and applied polynomial transformation to capture the pattern.

I also trained Decision Trees with different depths to understand how model complexity affects performance.

## Model Performance
- Polynomial R²: _____
- Tree (depth=2) R²: _____
- Tree (depth=5) R²: _____
- Tree (depth=20) R²: _____

## Observation
The polynomial model produces a smooth curve, while decision trees create step-like predictions. As depth increases, the tree becomes more complex and starts overfitting.

## Conceptual Reflection
A jittery model that hits every training point is worse because it memorizes the data instead of learning the underlying pattern.

Such a model performs poorly on new, unseen data because it is too sensitive to noise. A smoother model generalizes better and provides more reliable future predictions.

## Tools Used
- Python  
- NumPy  
- Matplotlib  
- Scikit-learn  