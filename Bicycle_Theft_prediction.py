# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 11:11:26 2021

@author : Vishal Patel
"""

import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import resample
from sklearn.metrics import accuracy_score,precision_score,recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import joblib


# Loading the file
filename = 'Bicycle_Thefts.csv'
path = 'C:/Users/Vishal pc/Downloads'
fullpath = os.path.join(path,filename)
df = pd.read_csv(fullpath)

df.head()
df.info()
df.columns

## range of data elements
df.describe().max() - df.describe().min()

## description
df.describe()

## finding null values
df.isnull().sum()

## plottimg histogram
df.hist(figsize=(10,12))
plt.show()

#Scatter Plot
sns.pairplot(df)

#Heat Map
sns.heatmap(df.corr(), cmap='coolwarm')

## dropping unnecessary columns
df.drop(['X','Y','OBJECTID','event_unique_id', 'ObjectId2'],inplace = True, axis=1)

#removing rows with unknown values in the Status class 
df = df[df["Status"]!="UNKNOWN"]

## applyimg label encoding for converting categorical data to numric values
from sklearn.preprocessing import LabelEncoder
label_enc = LabelEncoder()

df = df.apply(lambda col: label_enc.fit_transform(col.astype(str)), axis=0, result_type='expand')

## splitting into features and target
df_features=df[df.columns.difference(['Status'])] 
df_target=df['Status'] 

## balance the imablanced classes

df['Status'].value_counts()
df_majority = df[df.Status==1]
df_minority = df[df.Status==0]
df_features_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=24807,    # to match majority class                               
                                 random_state=123)
# Combine majority class with upsampled features class
df_upsampled = pd.concat([df_majority,df_features_upsampled])
 
# Display new class counts
df_upsampled.Status.value_counts()

y = df_upsampled.Status
x = df_upsampled.drop('Status', axis=1)

## splitting numeric and categorical features
numeric_features=['Occurrence_Year','Occurrence_DayOfMonth','Occurrence_DayOfYear','Occurrence_Hour','Report_Year', 'Report_DayOfMonth',
                  'Report_DayOfYear', 'Bike_Speed', 'Cost_of_Bike', 'Report_Hour']

cat_features = []
for i in df_features.columns:
    if i not in numeric_features:
        cat_features.append(i)
        

## creating pipeline
numeric_pipeline = Pipeline([
    ('impute',SimpleImputer(strategy='median')),
    ('standardization',StandardScaler())
])
category_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('OneHotEncoding',OneHotEncoder())
])

full_pipeline = ColumnTransformer([
    ('num_transformations',numeric_pipeline,numeric_features),
    ('cat_transformations',category_pipeline,cat_features)
])

x_transformed = full_pipeline.fit_transform(x)


## splitting data into train and test
x_train,x_test,y_train,y_test = train_test_split(x_transformed, y, test_size = 0.35)

### Logistic Regression Classifier 

#log_reg = LogisticRegression(solver="lbfgs", random_state=54)
#log_reg.fit(x_train,y_train)
#y_pred = log_reg.predict(x_test)
#print(f'Accuracy Score: {accuracy_score(y_test,y_pred)}')
def model_results(y_pred, y_test, x_test, model, estimator):
    
    # Evaluation:
    print("Estimator: ",estimator)
    # Accuracy score:
    print("Accuracy:", accuracy_score(y_test, y_pred)) 
    scores = cross_val_score(model, x_train, y_test, cv=5)
    print("Cross validation Score: ", scores.mean())
    print("Precision score:",precision_score(y_test, y_pred))
    print("Recall score:",recall_score(y_test, y_pred))
    print("F1 score:", f1_score(y_test, y_pred, average = "micro"), "\n")
    metrics.plot_roc_curve(model, x_test, y_test)  
    plt.show() 
    cm = (confusion_matrix(y_test,y_pred))
    sns.heatmap(cm,annot=True,fmt='g')
    

lr = LogisticRegression(max_iter = 1600,random_state=42)
lr.fit(x_train,y_train)
y_pred_lr = lr.predict(x_test)
model_results(y_pred_lr, y_test, x_test, lr, 'Logistic regression')

rfc = RandomForestClassifier()
rfc.fit(x_train,y_train)
y_pred_rfc = rfc.predict(x_test)
model_results(y_pred_rfc, y_test, x_test, rfc, 'Random Forest')

svm = SVC(C=0.1, kernel='linear')
svm.fit(x_train,y_train)
y_pred_svm = svm.predict(x_test)
model_results(y_pred_svm, y_test, x_test, svm, 'Support Vector Machine')

dtc = DecisionTreeClassifier(criterion='entropy', max_depth=42)
dtc.fit(x_train,y_train)
y_pred_dtc = dtc.predict(x_test)
model_results(y_pred_dtc, y_test,x_test, dtc, 'Decision Tree')

mlp = MLPClassifier(random_state=42)
mlp.fit(x_train,y_train)
y_pred_mlp = mlp.predict(x_test)
model_results(y_pred_mlp, y_test, mlp, 'MLP')

params = {'n_estimators':[50,100],
            'criterion':["gini","entropy"],
            'max_leaf_nodes':[40, 60],
            'min_samples_split':[5, 10, 20],
            'max_features':[20,40, 60]}
          
grid_search = GridSearchCV(
                    estimator=rfc,
                    param_grid=params,
                    scoring='accuracy',
                    cv=5,
                    refit=True,
                    verbose=3
)

grid_search.fit(x_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_estimator_)

grid_search.predict(x_test)

# save the model to disk
filename = 'finalized_model.sav'
joblib.dump(grid_search, filename)
# save the pipeline to disk
filename_pipeline = 'finalized_pipeline.sav'
joblib.dump(full_pipeline, filename_pipeline)

loaded_model = joblib.load(filename)
result = loaded_model.score(x_test, y_test)
print(result)


