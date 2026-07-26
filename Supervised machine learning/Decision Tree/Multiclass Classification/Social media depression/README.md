# Social Media Impact on Student Life - Decision Tree Classifier

## Overview
Multiclass classification project predicting the overall impact of social 
media usage on students' lives (Negative / Neutral / Positive), using a 
Decision Tree Classifier trained on survey data from 1,705 students to predict the overall impact of social media usage.

## Dataset
- **Source:** Social_media_impact_on_life.csv
- **Samples:** 1,705
- **Features:** Age, Gender, Academic_Level, Country, Avg_Daily_Usage_Hours, 
  Most_Used_Platform, Affects_Academic_Performance, Sleep_Hours_Per_Night
- **Target:** Overall_Impact (Negative / Neutral / Positive) — encoded as 0 / 1 / 2
- No missing values, no duplicates.

## Workflow
1. Exploratory Data Analysis (info, describe, count plots and histograms)
2. Encoding: label mapping for ordinal/binary columns (Gender, Academic_Level, 
   Affects_Academic_Performance, Overall_Impact), one-hot encoding for 
   Most_Used_Platform and Country
3. Train/test split (80/20) with `stratify=y` to preserve class balance
4. Model: `DecisionTreeClassifier(max_depth=5, random_state=42)`
5. Evaluation: accuracy, precision, recall, F1-score (weighted), confusion matrix
6. Model interpretation: tree visualization (`plot_tree`) and feature importance

## Data Leakage Investigation
An initial run that included `Mental_Health_Score` as a feature produced an 
unrealistically high accuracy (**97%**), with near-perfect F1-scores across 
all classes. Since `Overall_Impact` is a survey judgment closely tied to 
mental health, `Mental_Health_Score` was suspected of being derived from — 
or strongly encoding — the target itself.

After removing `Mental_Health_Score` from the feature set, accuracy dropped 
to **94%**, and recall for the "Neutral" class in particular fell to 0.74 
— a more realistic result for a 3-class problem based on survey data. 
Feature importance on the final model confirms no single feature dominates 
unnaturally, supporting that the leakage was resolved.

## Results
| Metric | Train | Test |
|--------|------:|-----:|
| Accuracy | 95.2% | 93.5% |

| Metric (Test set, weighted avg) | Score |
|----------------------------------|------:|
| Precision | 0.93 |
| Recall | 0.94 |
| F1-score | 0.93 |

| Class | Precision | Recall | F1-score |
|-------|----------:|-------:|---------:|
| Negative | 0.96 | 0.96 | 0.96 |
| Neutral  | 0.89 | 0.74 | 0.80 |
| Positive | 0.91 | 0.99 | 0.95 |

The "Neutral" class is the hardest to classify (lowest recall), which is 
expected — it's the middle ground between two more distinct classes.

## Feature Importance
| Feature | Importance |
|---------|-----------:|
| Sleep_Hours_Per_Night | 0.460 |
| Avg_Daily_Usage_Hours | 0.244 |
| Country_Other | 0.129 |
| Affects_Academic_Performance | 0.113 |
| Most_Used_Platform_LINE | 0.020 |
| Academic_Level | 0.010 |
| Age | 0.009 |
| Country_Ireland | 0.005 |
| Country_India | 0.005 |
| Country_Sri Lanka | 0.003 |

Sleep hours and daily usage hours together account for ~70% of the model's 
decisions — both are intuitively strong drivers of a student's overall 
wellbeing and academic outcomes.


## What I Learned
- Practiced multiclass classification (3 classes) with Decision Trees, 
  including `average='weighted'` for precision/recall/F1 in a multiclass setting.
- Identified and resolved a data leakage issue by critically examining 
  suspiciously high accuracy, rather than accepting it at face value.
- Learned to use feature importance as a diagnostic tool to confirm a model 
  is learning from genuine signal rather than a leaked feature.
