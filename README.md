# Decision Tree Machine Learning Portfolio

This repository contains three end-to-end machine learning projects built using **Decision Trees** across different real-world problem types:

- Regression (continuous prediction)
- Medical classification (clinical decision modeling)
- Behavioral classification (customer purchase intent)

The goal is to demonstrate practical Decision Tree usage, interpretability, and threshold-based decision strategies across domains.

---

## 📂 Projects Included

### 1️⃣ Housing Price Prediction (Regression)
**Folder:** `decision-tree-housing`  
- Task: Predict house prices (continuous target)  
- Model: DecisionTreeRegressor  
- Focus: Non-linear regression, overfitting control, residual analysis  

---

### 2️⃣ Heart Disease Detection (Medical Classification)
**Folder:** `decision-tree-Heart-disease`  
- Task: Predict presence of heart disease  
- Model: DecisionTreeClassifier  
- Features: Clinical and diagnostic variables  
- Threshold strategy: Precision-constrained (≥ 80%) via CV OOF  

This project demonstrates interpretable clinical decision rules.

---

### 3️⃣ Online Purchase Intent Prediction (Behavioral Classification)
**Folder:** `decision-tree-shoppers-intention`  
- Task: Predict whether a browsing session leads to purchase  
- Model: DecisionTreeClassifier  
- Dataset: Highly imbalanced session behavior  
- Threshold strategy: Recall-constrained (≥ 80%) via CV OOF  

This project models marketing-style decision tradeoffs.

---

## Why Decision Trees?

Decision Trees are powerful because they:

- Capture non-linear relationships
- Handle mixed data types
- Produce human-readable rules
- Allow decision-based threshold tuning
- Serve as the foundation for ensemble methods (Random Forest, Boosting)

---

## Techniques Demonstrated

Across the projects:

✔ Data cleaning & preprocessing  
✔ Regression and classification trees  
✔ Depth and leaf-size regularization  
✔ Cross-validation evaluation  
✔ Precision–Recall curve analysis  
✔ Threshold selection without test leakage  
✔ Confusion matrix interpretation  
✔ Business vs medical tradeoff reasoning  

---

## How to Run a Project

Each folder contains its own pipeline:

```bash
pip install -r requirements.txt
python src/preprocessing.py
python src/train.py
python src/evaluate.py
