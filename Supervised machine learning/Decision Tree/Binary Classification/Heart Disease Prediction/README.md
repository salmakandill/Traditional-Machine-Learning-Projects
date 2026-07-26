# Heart Disease Prediction - Decision Tree Classifier

## Overview
Binary classification project predicting the presence or absence of heart
disease using a Decision Tree Classifier, based on the UCI Heart Disease
dataset (270 records, 13 clinical features).

## Dataset
- **Source:** Heart_Disease_Prediction.csv (UCI-style Heart Disease dataset)
- **Samples:** 270
- **Features:** Age, Sex, Chest pain type, BP, Cholesterol, FBS over 120,
  EKG results, Max HR, Exercise angina, ST depression, Slope of ST,
  Number of vessels fluro, Thallium
- **Target:** Heart Disease (Presence / Absence) — encoded as 1 / 0
- No missing values, no duplicates.

## Workflow
1. Exploratory Data Analysis (info, describe, correlation heatmap, boxenplot, pairplot)
2. Target encoding (Presence → 1, Absence → 0)
3. Train/test split (80/20) with `stratify=y` to preserve class balance
4. Model: `DecisionTreeClassifier(max_depth=4, random_state=42)`
5. Evaluation: accuracy, precision, recall, F1-score, confusion matrix
6. Model interpretation: tree visualization (`plot_tree`) and feature importance

## Model Tuning Note
Initial run with `max_depth=5` showed a large gap between train accuracy
(~93%) and test accuracy (~70%), indicating overfitting. Reducing to
`max_depth=4` narrowed this gap and improved generalization.


## Results
| Metric | Train | Test |
|--------|------:|-----:|
| Accuracy | 91.7% | 79.6% |

| Metric (Test set) | Score |
|-------------------|------:|
| Precision | 0.74 |
| Recall | 0.83 |
| F1-score | 0.78 |

Recall was prioritized as a key metric given the medical context —
missing an actual heart disease case (false negative) is more costly
than a false alarm.

## Key Insight
Feature importance analysis shows **Chest pain type** had the strongest influence on the model's predictions.

## What I Learned
- Implemented a Decision Tree classifier for binary classification and learned how controlling `max_depth` helps reduce overfitting.
- Learned how to interpret Decision Trees using `plot_tree`, analyze feature importance, and evaluate model performance using Accuracy, Precision, Recall, F1-score, and the Confusion Matrix.
