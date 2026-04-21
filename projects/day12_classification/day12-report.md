# Day 12 Report – AI/ML Developer Track

**Name:** Vasudev Rawat  

## Technical Summary
Today, I implemented Logistic Regression for binary classification using the Breast Cancer dataset. The model predicts whether a tumor is malignant or benign.

I used StandardScaler to normalize features and evaluated performance using accuracy and a confusion matrix.

## Model Performance
- Accuracy: _______

## Confusion Matrix Insight
The confusion matrix shows how many predictions were correct and where the model made mistakes (false positives and false negatives).

## Probability vs Prediction
- predict() gives final class (0 or 1)
- predict_proba() gives probability of each class

## Conceptual Reflection
In a medical scenario, a False Negative is worse because it means a sick person is classified as healthy. This can delay treatment and may lead to serious consequences.

A False Positive is less harmful because further tests can confirm whether the person is actually sick.

## Tools Used
- Python  
- Scikit-learn  
- Seaborn  
- Matplotlib  