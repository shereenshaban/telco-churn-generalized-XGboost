# Telco Customer Churn Prediction

Predicting customer churn for a telecom company. The pipeline is built around avoiding data leakage at every step and choosing the final model based on **generalization (train/test gap)**, not just the highest raw score.

## Dataset

`Telco_Customer_Churn.xlsx` — customer-level data including demographics, account information (tenure, contract type, charges), and subscribed services. Target: **Churn Value** (0 = stayed, 1 = churned).

Columns dropped to prevent target leakage: `Churn Label`, `Churn Score`, `Churn Reason`, `Churn Value` (kept only as target), `CustomerID`, `Count`.

## Pipeline

1. **Load & split** — 80/20 train/test split, stratified on the target (5,634 train / 1,409 test).
2. **Missing value handling** — median imputation, statistics computed on the training set only.
3. **Feature engineering** — added:
   - `Avg_Monthly_Spend` = Total Charges / (Tenure Months + 1)
   - `Is_New_Customer` = Tenure Months < 12
   - `Is_Loyal_Customer` = Tenure Months ≥ 48
   - `CLTV_per_Month` = CLTV / (Tenure Months + 1)
4. **Outlier capping** — numeric features capped at the 5th/95th percentile, thresholds derived from the training set only.
5. **High-cardinality check** — flags ID-like columns (unique ratio ≥ 90%), e.g. `Total Charges`, `CLTV`.
6. **Drop geo/fixed columns** — `Country`, `State`, `Lat Long`, `City`, `Zip Code`, `Latitude`, `Longitude` (single-value or location columns not useful for a generalizable model).
7. **One-hot encoding** — categorical features encoded, train/test columns aligned.
8. **Correlation check** — flags feature pairs with |correlation| > 0.8 (mostly expected redundancy, e.g. `Internet Service_No` with the "No internet service" dummy columns).
9. **Z-score normalization** — mean/std computed on the training set only.

## Experiments

| # | Model | Training AUC | CV AUC (5-fold) | Test AUC | Train–Test/CV Gap |
|---|---|---|---|---|---|
| 1 | Logistic Regression (L2, C=1.0) | 0.8665 | 0.8625 ± 0.0130 | – | ~0.004 |
| 2 | PCA (30 comp.) + Logistic Regression | 0.8665 | 0.8625 ± 0.0130 | – | ~0.004 |
| 3 | Random Forest | 0.9268 | 0.8553 ± 0.0121 | – | **~0.072** |
| 4 | **XGBoost** | 0.8721 | 0.8619 ± 0.0122 | **0.8527** | **~0.010** |
| 5 | LightGBM (regularized manually) | 0.9003 | – | 0.8576 | ~0.043 |
| 6 | LightGBM + RandomizedSearchCV (50 candidates, 5-fold) | – | 0.8654 | 0.8559 | ~0.010 |

## Why XGBoost was chosen as the final model

The goal was not just the highest AUC on a single split, but a model that **generalizes consistently** across training, cross-validation, and the held-out test set.

- **Random Forest** reached the highest training AUC (0.9268) but its CV AUC dropped to 0.8553 — a gap of ~0.07, a clear sign of overfitting to the training data.
- **LightGBM**, even after manual regularization (limited depth, `min_child_samples`, L1/L2 penalties), still showed a training/test gap of ~0.043 — better than the raw Random Forest, but still overfitting more than desired.
- **LightGBM tuned with `RandomizedSearchCV`** (50 candidates × 5-fold CV) closed that gap (CV 0.8654 vs. Test 0.8559, gap ~0.01), but only after being forced into a very conservative parameter region (`max_depth=3`, `num_leaves=7`, `learning_rate=0.01`) — meaning the model's natural tendency is to overfit this dataset, and it needs heavy tuning just to behave. It also did not exceed XGBoost's performance after tuning.
- **XGBoost**, using reasonably conservative default-ish parameters (`max_depth=3`, `learning_rate=0.03`, `subsample=0.7`, `colsample_bytree=0.7`) without any extra tuning step, already achieved a small, stable gap between Training AUC (0.8721), CV AUC (0.8619 ± 0.0122), and Test AUC (0.8527) — around 0.01–0.02 throughout. This is the most consistent generalization behavior across all experiments.

**Conclusion:** XGBoost was selected as the final model because it delivers competitive AUC while showing the least evidence of overfitting, without needing an extensive hyperparameter search to get there. That makes it the more trustworthy choice for predicting churn on genuinely unseen customers, not just the specific test split used here.

**Final result (held-out test set):**
```
Training AUC-ROC: 0.8721
CV AUC-ROC:        0.8619 ± 0.0122
Test AUC-ROC:      0.8527
```

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
torch
openpyxl
```

Install with:

```bash
pip install -r requirements.txt
```

## Usage

1. Place `Telco_Customer_Churn.xlsx` in the project root.
2. Run the notebook `Telco_customer_Churn_notebook.ipynb` top to bottom.

## Next steps

- Try ensembling XGBoost with Logistic Regression (L1) — the two capture different patterns and a simple average of probabilities may push AUC slightly higher without adding overfitting risk.
- Inspect XGBoost feature importances to prune low-value one-hot columns (e.g. the perfectly correlated "No internet service" dummy columns) and simplify the feature set.
- Test threshold tuning (instead of the default 0.5) against a business cost matrix (cost of missed churner vs. cost of false alarm), since AUC alone doesn't reflect deployment costs.

## License

Add a license of your choice (e.g., MIT) if this repo is public.
