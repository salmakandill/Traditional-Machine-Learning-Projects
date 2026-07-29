# Heart Disease Prediction - Random Forest Classifier

## Overview
Binary classification project predicting the presence or absence of heart
disease using a Random Forest Classifier, based on the UCI Heart Disease
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
4. Model:
   `RandomForestClassifier(
   n_estimators=100,
   max_depth=4,
   max_features='sqrt',
   class_weight='balanced',
   random_state=42
   )`
5. Evaluation: accuracy, precision, recall, F1-score, confusion matrix
6. Model interpretation: feature importance

## Model Tuning
The initial Random Forest achieved high training accuracy but showed room
for better generalization. Limiting the tree depth (`max_depth=4`) and
using `max_features='sqrt'` reduced overfitting, while
`class_weight='balanced'` ensured both classes contributed equally during
training.

## Results

| Metric | Train | Test |
|--------|------:|-----:|
| Accuracy | 93.5% | 83.3% |

| Metric (Test set) | Score |
|-------------------|------:|
| Precision | 0.80 |
| Recall | 0.83 |
| F1-score | 0.82 |

Recall remains an important metric in this medical application because
missing a patient with heart disease (false negative) is more costly than
incorrectly predicting disease for a healthy patient.

## Feature Importance

Feature importance indicates that **Chest pain type**, **Thallium**, **Number of vessels fluro**, and **ST depression** are among the strongest predictors, aligning with known clinical indicators used in heart disease diagnosis.

## What I Learned

- Implemented an ensemble learning algorithm (Random Forest) for binary classification.
- Learned how Random Forest reduces overfitting by combining predictions from multiple decision trees.
- Practiced tuning key hyperparameters such as `n_estimators`, `max_depth`, `max_features`, and `class_weight`.
- Compared Decision Tree and Random Forest, observing that Random Forest provided better generalization while being less prone to overfitting.