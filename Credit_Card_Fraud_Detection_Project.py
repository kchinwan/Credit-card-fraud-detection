#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

import sklearn
import scipy

from pylab import rcParams
rcParams['figure.figsize'] = 16,12
RANDOM_SEED = 101
LABELS = ['Normal' , 'Fraud']

# import dataset
data = pd.read_csv('CreditCard.csv' , sep = ',')
data.head()


# Preprocessing
#step 1: check missing data
data.isnull().sum()


# step 5 : check imbalance dataset
data['Class'].value_counts()     # there is a imbalance dataset

# but here we cannot handle imbalance dataset bcoz this is domain specific problem . 


# step 6: EDA

count_classes = pd.value_counts(data['Class'], sort=True)
count_classes.plot(kind = 'bar' , rot = 0)
plt.title('Transaction class distribution')
plt.xticks(range(2), LABELS)
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()


fraud = data[data['Class'] == 1]
normal = data[data['Class'] == 0]

print(fraud.shape , normal.shape)


normal.Amount.describe()


fraud.Amount.describe()



f,(ax1 , ax2) = plt.subplots(2,1 , sharex = True)
f.suptitle("Amount per transaction by class")
bins = 50
ax1.hist(fraud.Amount, bins=bins)
ax1.set_title('Fraud')

ax2.hist(normal.Amount, bins = bins)
ax2.set_title('normal')

plt.xlabel("Amount ($)")
plt.ylabel("No. of Transaction")

plt.xlim(0,20000)
plt.yscale('log')
plt.show()



f,(ax1 , ax2) = plt.subplots(2,1 , sharex = True)
f.suptitle("Time per transaction by class")

ax1.scatter(fraud.Time , fraud.Amount)
ax1.set_title('Fraud')

ax2.scatter(normal.Time , normal.Amount)
ax2.set_title('normal')

plt.xlabel("Time (s)")
plt.ylabel("Amount")

plt.show()


# taking only 10% of the data for model building

data1 = data.sample(frac = 0.1, random_state = 1)
data1.shape


fraud = data[data['Class']==1]
valid = data[data['Class']==0]


# correlation part:Heat map
corrmat = data1.corr()
top_corr_feature = corrmat.index
plt.figure(figsize=(30,30))
g = sns.heatmap(data1[top_corr_feature].corr() , annot = True , cmap = 'coolwarm')


columns = data1.columns.tolist()
columns

# split the data into target variable and feature variables

x = data1.iloc[:,0:31]
y = data1['Class']
state = np.random.RandomState(101)
x_outlier = state.uniform(low = 0, high = 1, size = (x.shape[0], x.shape[1]))
print(x.shape)
print(y.shape)


# split the data into train test for model building

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.70, random_state=119)


# ## Logistic Regression


from sklearn.linear_model import LogisticRegression
logit = LogisticRegression()
logit.fit(x_train,y_train)


y_pred_train_logit = logit.predict(x_train)
y_pred_test_logit = logit.predict(x_test)


from sklearn.metrics import accuracy_score , confusion_matrix, classification_report

print(classification_report(y_train ,y_pred_train_logit))
print("******"*50)
print(classification_report(y_test ,y_pred_test_logit))


print(confusion_matrix(y_train ,y_pred_train_logit))
print("******"*50)
print(confusion_matrix(y_test ,y_pred_test_logit))


print(accuracy_score(y_train ,y_pred_train_logit))
print("******"*50)
print(accuracy_score(y_test ,y_pred_test_logit))


# ## Decision Tree


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion = 'entropy')
dtree.fit(x_train , y_train)



y_pred_train_dtree = dtree.predict(x_train)
y_pred_test_dtree = dtree.predict(x_test)


print(accuracy_score(y_train ,y_pred_train_dtree))
print("******"*50)
print(accuracy_score(y_test ,y_pred_test_dtree))


# ## Random Forest`


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 500 , criterion = 'entropy')
rfc.fit(x_train , y_train)



y_pred_train_rfc = rfc.predict(x_train)
y_pred_test_rfc = rfc.predict(x_test)


print(accuracy_score(y_train ,y_pred_train_rfc))
print("******"*50)
print(accuracy_score(y_test ,y_pred_test_rfc))


# ## XGBoost


get_ipython().system('pip install xgboost')



from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(x_train , y_train)


y_pred_train_xgb = xgb.predict(x_train)
y_pred_test_xgb = xgb.predict(x_test)


print(accuracy_score(y_train ,y_pred_train_xgb))
print("******"*50)
print(accuracy_score(y_test ,y_pred_test_xgb))



print(classification_report(y_train ,y_pred_train_xgb))
print("******"*50)
print(classification_report(y_test ,y_pred_test_xgb))


# ## Stacking Classifier


get_ipython().system('pip install mlxtend')



from mlxtend.classifier import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


from sklearn.model_selection import cross_val_score

knn = KNeighborsClassifier(n_neighbors = 5)
gnb = GaussianNB()
rfc_c = RandomForestClassifier()
lg = LogisticRegression()
sc = StackingClassifier(classifiers = [knn , gnb , rfc_c] , meta_classifier = lg)

print('3-fold cross validation : \n')

for clf , label in zip([knn,gnb,rfc_c,sc],['KNN','Naive Bayes','Random Forest','Stacking Classifier']):
    scores = cross_val_score(clf , x , y , cv=3,scoring = 'accuracy')
    print("Accuracy : %0.2f (+/-%0.2f)[%s]" % (scores.mean() , scores.std() , label))



# ## Isolation Forest

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM



Classification = {'Isolation Forest' :IsolationForest(n_estimators = 100 , max_samples = len(x) , 
                                                                contamination = len(fraud)/float(len(valid))),
                  
                 'Local outlier Factor' : LocalOutlierFactor(n_neighbors = 20 , 
                                                                 contamination = len(fraud)/float(len(valid))),
                  
                 'One Class SVM' : OneClassSVM()}


n_outlier = len(fraud)

for i , (clf_name , clf) in enumerate(Classification.items()):
    if clf_name == 'Local outlier Factor':
        y_pred = clf.fit_predict(x)
        score_prediction = clf.negative_outlier_factor_
    elif clf_name == 'One Class SVM':
        clf.fit(x)
        y_pred = clf.predict(x)
    else:
        clf.fit(x)
        score_prediction = clf.decision_function(x)
        y_pred = clf.predict(x)
        
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    n_error = (y_pred != y).sum()
    
    print("{} : {}".format(clf_name , n_error))
    print("Accuracy Score : ")
    print(accuracy_score(y , y_pred))
    print("Classification Report : ")
    print(classification_report(y,y_pred))
    


# # Hence Isolation Forest Model will give best answer 





