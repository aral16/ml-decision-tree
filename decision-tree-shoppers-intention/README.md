# Online Purchase Intent Prediction using Decision Trees

## Problem
E-commerce platforms aim to identify user sessions that are likely to convert into purchases.  
The objective of this project is to predict whether a browsing session will result in revenue.

## Dataset
Source: UCI Online Shoppers Purchasing Intention  
- 12,330 sessions  
- Mix of behavioral and categorical features (pages visited, bounce rate, month, visitor type, etc.)  
- Target:
  - Revenue = True → purchase occurred
  - Revenue = False → no purchase

The dataset is highly imbalanced (~15% positive sessions).

## Why Decision Trees?
Decision Trees are ideal for:
- Behavioral interaction modeling
- Mixed categorical/numerical features
- Human-readable conversion rules

Example logic captured by the tree:
> IF visitor is returning AND product pages viewed > threshold → high purchase likelihood

## Model
DecisionTreeClassifier  
- max_depth = 3 (shallow, interpretable)  
- min_samples_leaf = 10  
- Stratified 5-fold Cross-Validation

## Threshold Strategy
Threshold selected using CV out-of-fold predictions (no test leakage).

Goal:
- Maintain Recall ≥ 80% (capture most buyers)
- Maximize Precision under that constraint

Chosen threshold: 0.348


---


## How to Run
```bash
pip install -r requirements.txt
python src/preprocessing.py
python src/train.py
python src/evaluate.py
