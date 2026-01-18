# Decision-Tree — House Prices (Ames, Iowa)

## Problem
Predict residential sale prices using structural, demographic, and location-based housing features.

## Data
Kaggle — **House Prices: Advanced Regression Techniques**  
Training file: `data/raw/train.csv`

# 📥 How to Get the Dataset (Manual Kaggle Download)

Kaggle datasets cannot be redistributed, so you must download the data manually.

---

## Step-by-step Instructions

1. Go to the Kaggle competition page:  
   👉 https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

2. Log into your Kaggle account.

3. Click the **Download** button (top-right of the page).

4. Extract the downloaded ZIP file.

5. Copy the following files into the project directory:



## ML Task
Decision Tree

---

## Approach
1. Split data into Train/Test (20% test held out)
2. Log-transform target (`log1p(SalePrice)`) to reduce skew and stabilize variance
3. Preprocess features:
   - Numeric: median imputation + standard scaling
   - Categorical: most-frequent imputation + one-hot encoding
4. Train: Decision-Tree
5. Evaluate:
   - 5-Fold Cross-Validation (log space)
   - Test metrics (converted back to original dollar scale)
6. Residual diagnostics + coefficient interpretation

---

## Model Evaluation & Analysis

### Training Strategy
The model was trained using a Decision Tree Regressor on the Kaggle House Prices dataset.  
To reduce skew and stabilize variance, the target (`SalePrice`) was log-transformed using `log1p`.

Evaluation was performed using 5-fold cross-validation on the training set (log space), followed by final testing on a held-out test set (original dollar scale).

---

## Cross-Validation Results (Log Space)

| Metric | Mean | Std |
|--------|------|------|
| MAE | 0.137 | 0.0029 |
| RMSE | 0.190 | 0.0057 |
| R² | 0.762 | 0.0287 |

### Interpretation
- The model is stable across folds (low std).
- R² indicates the tree captures some structure.
- However, MAE is noticeably worse than the linear regression baseline (check linear-regression-housing repo) from the previous project.
- However, MAE is noticeably worse than the linear regression baseline from my previous project (see the README here):  
👉 [Linear Regression Housing Repo](https://github.com/aral16/ml-linear-regression.git)


This suggests that while the tree learns patterns, it does not model proportional price relationships efficiently.

---

## Test Set Results (Original Dollar Scale)

| Metric | Value |
|--------|-------|
| MAE | \$24,265 |
| RMSE | \$40,234 |
| R² | 0.768 |

### Interpretation
Performance on unseen data drops significantly compared to the linear regression model (R² ≈ 0.93).

The Decision Tree struggles to generalize because:

- The dataset contains hundreds of one-hot encoded categorical features.
- Trees split feature space into many small regions, causing fragmentation.
- High-dimensional sparse matrices reduce the effectiveness of single-tree models.
- Predictions become piecewise constant rather than smoothly varying with features.

This leads to large errors, particularly for expensive houses.

---
---

## Residual Diagnostics (Original Scale)

To better understand model behavior beyond aggregate metrics, residual analysis was performed on the held-out test set.

Residual = (True Price − Predicted Price)

### 1) Residual Distribution

The histogram shows:

- Most residuals are concentrated near zero → the model predicts typical houses reasonably well.
- A long right tail is present → several houses are heavily under-predicted.
- Extreme positive residuals (> \$100k) indicate the tree fails on high-value properties.
- The distribution is not symmetric → systematic bias exists for expensive homes.

This asymmetry is a classic sign that a single decision tree cannot model smooth price variation across complex feature interactions.

---

### 2) Residuals vs Predictions

This plot reveals a strong structural issue:

- Residual variance increases sharply as predicted price increases.
- Large positive residuals dominate high predicted ranges → expensive houses are consistently underestimated.
- Residuals appear in vertical bands → tree outputs piecewise-constant predictions (typical tree behavior).
- No random scatter pattern → the model is missing important structure.

This confirms:

> The Decision Tree is not interpolating continuously; it is partitioning the space into coarse regions and averaging prices within leaves.

Such behavior leads to poor generalization on high-dimensional sparse tabular data like Ames Housing.

---

## Residual Analysis Conclusion

Residual diagnostics reinforce the metric-based findings:

- The model performs acceptably on mid-range houses.
- Severe underprediction occurs for high-price properties.
- Error variance grows with price (heteroskedasticity).
- Piecewise prediction bands highlight the limitations of single-tree models.

These patterns clearly show that while decision trees capture non-linear splits, they lack the smoothness and robustness needed for this dataset. This motivates transitioning to ensemble tree methods (Random Forest / Gradient Boosting) to reduce variance and improve generalization.

---

## Key Takeaways
- Decision Trees are powerful for non-linear modeling but perform poorly in very wide, sparse feature spaces.
- Linear regression handled one-hot features better because it assigns independent coefficients without fragmenting the data.
- The tree underfits important continuous relationships and fails to generalize well.

---

## Conclusion
The Decision Tree model highlights the limitations of single-tree regression on high-dimensional tabular data.  
These results motivate the next step: ensemble tree models such as Random Forest and Gradient Boosting, which are designed to overcome these weaknesses and significantly improve predictive performance.

---

## How to Run
```bash
pip install -r requirements.txt
python src/preprocessing.py
python src/train.py
python src/evaluate.py
