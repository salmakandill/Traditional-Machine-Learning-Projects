#Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import (DecisionTreeClassifier , plot_tree)
from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

#Data preprocessing
df = pd.read_csv(r'data\Heart_Disease_Prediction.csv')

print(df.info())
print(df.head())
print(df.describe())
print(df.isna().sum())
print(f'Dataset shape: {df.shape}')
print(f'Dataset Columns :{df.columns}')
print(df.duplicated().sum())

df['Heart Disease']=df['Heart Disease'].map({
    'Presence' : 1,
    'Absence'  : 0
})

X= df.drop('Heart Disease',axis=1)
y=df['Heart Disease']


#Data visualisation
sns.boxenplot(
    x='Heart Disease',
    y='Cholesterol',
    data=df
)
plt.savefig("images/outliers.png", dpi=300)
plt.show()

plt.figure(figsize=(12,10))
sns.heatmap(df.corr(),annot=True)
plt.savefig("images/correlation.png", dpi=300)
plt.show()

sns.pairplot( 
    df[
        [
            'Age',
            'BP',
            'Cholesterol',
            'Max HR',
            'Heart Disease'
        ]
    ],
    hue='Heart Disease',
        diag_kind='hist'
    )
plt.savefig("images/features.png", dpi=300)
plt.show()


#model training
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=True,random_state=42 , stratify=y)

model=DecisionTreeClassifier(
    random_state=42,
    max_depth=4
    )

model.fit(X_train,y_train)

train_accuracy = model.score(X_train,y_train)
test_accuracy = model.score(X_test,y_test)

print('train_accuracy : ',train_accuracy)
print('test_accuracy : ',test_accuracy)

#model predection
y_pred=model.predict(X_test)


#evaluation
accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)
f1score=f1_score(y_test,y_pred)
classificationreport = classification_report(y_test,y_pred)

print(f'Accuracy = {accuracy:.2f}')
print(f'Precision = {precision:.2f}')
print(f'Recall = {recall:.2f}')
print(f'F1_Score = {f1score:.2f}')
print(f'Classification Report = {classificationreport}')

results=pd.DataFrame({
    'Actual Heart Disease ' : y_test.values,
    'Predicted Heart Disease' : y_pred
})

print(results.head(10))


#Confusion matrix
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(
    cm,
    annot= True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Absence','Presence'],
    yticklabels=['Absence','Presence']
)
plt.xlabel('predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig("images/confusion_matrix.png", dpi=300)
plt.show()


# Tree Visualization
plt.figure(figsize=(20,10))
plot_tree(
    model,
    feature_names=list(X.columns),
    class_names=['Absence', 'Presence'],
    filled=True,
    fontsize=10,
    rounded=True
)
plt.savefig("images/tree_plot.png", dpi=300)
plt.show()


#feature importances
importance = pd.Series(model.feature_importances_,index=X.columns)
importance = importance.sort_values(ascending=False)

plt.figure(figsize=(10,6))
importance.plot(kind='barh' , color ='teal')

plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.savefig("images/feature_importances.png", dpi=300)
plt.show()