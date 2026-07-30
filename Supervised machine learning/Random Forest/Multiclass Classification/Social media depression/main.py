#Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Data Loading & Exploration
df=pd.read_csv(r'data\Social_media_impact_on_life.csv')

print(df.info())
print(df.head())
print(df.describe())
print(df.isna().sum())
print(f'Dataset shape: {df.shape}')
print(f'Dataset Columns :{df.columns}')
print(df.duplicated().sum())

#Data preprocessing
df['Gender'] = df['Gender'].map({
    'Male' : 0,
    'Female' : 1
})

df['Affects_Academic_Performance']=df['Affects_Academic_Performance'].map({
    'No': 0 ,
    'Yes': 1
})

df['Academic_Level']=df['Academic_Level'].map({
    'High School': 0 ,
    'Undergraduate': 1,
    'Graduate':2
})


df['Overall_Impact']=df['Overall_Impact'].map({
    'Negative' :0,
    'Neutral' :1,
    'Positive' :2
})

df=pd.get_dummies(
    df,
    columns=['Most_Used_Platform','Country'],
    drop_first=True,
    dtype=int
)

X=df.drop(['Overall_Impact','Student_ID','Mental_Health_Score'],axis=1)
y=df['Overall_Impact']


#Data visualisation
sns.countplot(
    x='Gender',
    hue='Gender',
    data=df,
    palette='flare',
    legend=False
)
plt.savefig("images/gender.png", dpi=300)
plt.show()



sns.histplot(
    df['Avg_Daily_Usage_Hours'],
    bins=10,
    kde=True
)
plt.title('Daily Social Media Hours')
plt.savefig("images/Avg_Daily_Usage_Hours.png", dpi=300)
plt.show()

sns.countplot(
    x='Overall_Impact',
    hue='Overall_Impact',
    data=df,
    palette='viridis',
    legend=False
)

plt.title('Overall Impact Distribution')
plt.savefig("images/Overall_Impact_Distribution.png", dpi=300)
plt.show()


#Model Training
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=True,random_state=42,stratify=y)

model=RandomForestClassifier(
    random_state=42,
    max_depth=5,
    n_estimators= 100,
    class_weight='balanced'
)
model.fit(X_train,y_train)

train_accuracy=model.score(X_train,y_train)
test_accuracy=model.score(X_test,y_test)

print(f'Train Accuracy = {train_accuracy:.2f}')
print(f'Test Accuracy = {test_accuracy:.2f}')


#Model Prediction
y_pred=model.predict(X_test)
pred_probabilities = model.predict_proba(X_test)


#Evaluation
accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred,average='weighted')
recall=recall_score(y_test,y_pred,average='weighted')
f1score=f1_score(y_test,y_pred,average='weighted')
classificationreport = classification_report(y_test,y_pred)


print('Predection Probabilities',pred_probabilities[:5])
print(f'Accuracy = {accuracy:.2f}')
print(f'Precision = {precision:.2f}')
print(f'Recall = {recall:.2f}')
print(f'F1_Score = {f1score:.2f}')
print(f'Classification Report = {classificationreport}')



results=pd.DataFrame({
    'Actual OverAll Imapact ' : y_test.values,
    'Predicted OverAll Imapact ' : y_pred
})

print(results.head(10))


#Confusion matrix
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Negative','Neutral','Positive'],
    yticklabels=['Negative','Neutral','Positive']
)
plt.xlabel('predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig("images/confusion_matrix.png", dpi=300)
plt.show()


#Feature Importances
importance = pd.Series(model.feature_importances_,index=X.columns)
importance = importance.sort_values(ascending=False).head(10)

print('Feature Importances : ' ,importance)

plt.figure(figsize=(15,8))
importance.plot(kind='barh', color='teal')
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.savefig("images/feature_importances.png", dpi=300)
plt.show()