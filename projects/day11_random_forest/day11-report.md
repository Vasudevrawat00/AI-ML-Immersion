# Day 11 Report – AI/ML Developer Track

**Name:** Vasudev Rawat  


## Technical Summary
Today, I implemented Random Forest Regression on the California Housing dataset. Random Forest is an ensemble method that combines multiple decision trees to improve accuracy and reduce variance.

I also analyzed feature importance to understand which factors influence house prices the most.

## Model Performance
- Random Forest R²: _______

## Feature Importance
Top features affecting predictions:
- _______
- _______
- _______

## Experiment: Tree Count Test
| Trees | R² Score | Time (sec) |
|------|--------|-----------|
| 10   | ______ | ______ |
| 50   | ______ | ______ |
| 200  | ______ | ______ |

## Observation
As the number of trees increases, the model generally becomes more stable and accurate. However, after a certain point, the improvement becomes very small while training time increases significantly.

## Conceptual Reflection
Increasing the number of trees does not improve accuracy forever. After a certain point, we reach diminishing returns, where adding more trees increases computation time but gives only minor performance improvement.

## Tools Used
- Python  
- Scikit-learn  
- Pandas  
- Matplotlib  