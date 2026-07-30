# Social Media Impact on Student Life - Random Forest Classifier

## Overview
This project applies a **Random Forest Classifier** to predict the overall impact of social media usage on students' lives (**Negative / Neutral / Positive**) using survey data from **1,705 students**.

Unlike the previous Decision Tree model, Random Forest combines the predictions of multiple decision trees to improve generalization and reduce overfitting.

---

## Dataset

- **Source:** `Social_media_impact_on_life.csv`
- **Samples:** 1,705
- **Target:** `Overall_Impact`
  - Negative → 0
  - Neutral → 1
  - Positive → 2

### Features
- Age
- Gender
- Academic_Level
- Country
- Avg_Daily_Usage_Hours
- Most_Used_Platform
- Affects_Academic_Performance
- Sleep_Hours_Per_Night

**Removed Features**
- `Student_ID` (identifier)
- `Mental_Health_Score` (removed to avoid data leakage)

The dataset contains:
- No missing values
- No duplicate records

---

## Model

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    class_weight="balanced",
    random_state=42
)
```

### Parameters

- **n_estimators = 100**
  - Uses 100 decision trees.

- **max_depth = 5**
  - Limits tree depth to reduce overfitting.

- **class_weight = "balanced"**
  - Gives more balanced importance to each class during training.

- **random_state = 42**
  - Ensures reproducible results.

---

## Data Leakage Investigation

During the Decision Tree project, including `Mental_Health_Score` resulted in an unrealistically high accuracy because it strongly reflected the target variable (`Overall_Impact`).

To avoid data leakage, the feature was removed before training the Random Forest model.

---

## Results

| Metric | Train | Test |
|--------|------:|-----:|
| Accuracy | **95%** | **91%** |

### Test Metrics

| Metric | Score |
|--------|------:|
| Accuracy | **0.91** |
| Precision | **0.92** |
| Recall | **0.91** |
| F1-score | **0.91** |

### Classification Report

| Class | Precision | Recall | F1-score |
|------|----------:|-------:|---------:|
| Negative | 0.97 | 0.94 | 0.95 |
| Neutral | 0.89 | 0.74 | 0.80 |
| Positive | 0.83 | 0.96 | 0.89 |

The **Neutral** class remains the most difficult to classify, showing the lowest recall. This is expected since it represents the boundary between positive and negative outcomes.

---

## Feature Importance

The most influential features are:

| Feature | Importance |
|---------|-----------:|
| Sleep_Hours_Per_Night | Highest |
| Avg_Daily_Usage_Hours | High |
| Affects_Academic_Performance | Moderate |
| Country_Other | Moderate |
| Country_Denmark | Lower |

Sleep duration and daily social media usage remain the strongest predictors of a student's overall social media impact.

---

## Visualizations

The project includes:

- Gender distribution
- Daily social media usage distribution
- Overall impact distribution
- Confusion Matrix
- Feature Importance

---

## Comparison with Decision Tree

| Model | Train Accuracy | Test Accuracy |
|------|---------------:|--------------:|
| Decision Tree | 95% | 94% |
| Random Forest | 95% | 91% |

Although Random Forest is generally more robust than a single Decision Tree, on this dataset the Decision Tree achieved slightly higher test accuracy. This may be due to the dataset's relatively small size and the simplicity of the underlying decision boundaries.

---

## What I Learned

- Learned how Random Forest improves Decision Trees using **Bagging** and **majority voting**.
- Understood the purpose of `n_estimators`, `max_depth`, and `class_weight`.
- Practiced multiclass classification with ensemble methods.
- Compared Random Forest performance with a Decision Tree on the same dataset.
- Reinforced the importance of detecting and removing **data leakage** before training.
- Used Feature Importance to interpret which variables contributed most to the model's predictions.