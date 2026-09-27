# Fraud Detection using Random Forest

A machine learning project applying a Random Forest classifier to the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset, aiming to identify fraudulent credit card transactions.

Prepared as part of the MSc. Computer Engineering, Cybersecurity and Artificial Intelligence program. See [`AI project.pdf`](AI%20project.pdf) for the full report.

## Overview

Rule-based fraud detection systems suffer from high false positive rates and can't adapt to evolving fraud patterns. This project evaluates whether a Random Forest classifier can outperform simple heuristics by learning non-linear patterns from transaction and identity data.

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) — transaction and identity data provided by Vesta Corporation, split into separate transaction/identity tables for train and test sets. CSV files are not included in this repo (see `.gitignore`); download them from Kaggle and place them one directory above `source notebooks/`.

## Methodology

1. **Preprocessing** ([`preprocessing.ipynb`](source%20notebooks/preprocessing.ipynb))
   - Merge transaction and identity tables on `TransactionID` (right join, keeping all transactions)
   - Drop columns with more than 80% missing values
   - Impute remaining missing values (mode for categorical, mean for numerical columns), fit on train only to avoid leakage
   - Encode categorical columns with `LabelEncoder`, mapping unseen test categories to `-1`
   - Random under-sampling of the majority (non-fraud) class on the training set only, to address class imbalance while keeping the test set realistic

2. **Modeling** ([`classification.ipynb`](source%20notebooks/classification.ipynb))
   - Baseline: single Decision Tree
   - `RandomForestClassifier` (entropy criterion) tuned via `GridSearchCV` over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, scored on ROC AUC with 5-fold CV
   - Final model re-evaluated with `StratifiedKFold` (5 folds) to preserve class distribution across folds

## Results

| Model | ROC AUC |
|---|---|
| Decision Tree (baseline) | 0.7955 |
| Random Forest (tuned, 5-fold CV) | 0.9273 ± 0.0015 |

Best hyperparameters: `n_estimators=400`, `max_depth=40`, `min_samples_split=2`, `min_samples_leaf=1`.

Final submission to the Kaggle competition scored **0.9165** (public leaderboard) and **0.8936** (private leaderboard).

## Repository structure

```
.
├── AI project.pdf                     # Full written report
└── source notebooks/
    ├── preprocessing.ipynb            # Data merging, cleaning, encoding, under-sampling
    └── classification.ipynb           # Decision tree baseline, Random Forest tuning & evaluation
```

## Requirements

- Python 3
- pandas, scikit-learn, imbalanced-learn (`imblearn`), matplotlib

## Notes

Gradient boosting methods (XGBoost, LightGBM, CatBoost) and deep learning approaches (e.g. autoencoders for anomaly detection) are known to outperform Random Forest on this type of task and are natural directions for further work.
