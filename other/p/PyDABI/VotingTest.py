#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:36:02 2019

@author: zahid
"""

import pandas as pd

from sklearn.ensemble import AdaBoostClassifier #Import adabost
from sklearn import svm #Import SVM
from sklearn.linear_model import LogisticRegression #Import Logistic regression Model
from sklearn.ensemble import RandomForestClassifier #Import Random Forest Model
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.ensemble import VotingClassifier
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix # confusion_matrix for model evaluation

data = pd.read_csv('data/creditcard.csv')

feature_cols = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8',
                'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15',
                'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22',
                'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
X = data[feature_cols] # Features
y = data['Class'] # Target variable


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test



kfold = model_selection.KFold(n_splits=10, random_state=1)

# create the sub models
estimators = []

model1 = LogisticRegression()
estimators.append(('logistic', model1))

model2 = RandomForestClassifier(n_estimators=100)
estimators.append(('rf', model2))

model3 = svm.SVC(kernel='linear')
estimators.append(('svm', model3))

model4 = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1)
estimators.append(('ada', model4))

# create the ensemble model
ensemble = VotingClassifier(estimators)

ensemble.fit(X_train,y_train)

y_pred=ensemble.predict(X_test)

# Model confusion_matrix
print(confusion_matrix(y_test, y_pred))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
total = tn+fp+fn+tp

# AUC 
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
auc = metrics.auc(fpr, tpr)

acc = metrics.accuracy_score(y_test, y_pred)
pre = metrics.precision_score(y_test, y_pred, average='binary')
rec = metrics.recall_score(y_test, y_pred, average='binary')
f1s = metrics.f1_score(y_test, y_pred, average='binary')
tnp = tn/total
fpp = fp/total
fnp = fn/total
tpp = tp/total

print("Accuracy:\t",acc)
print("True positive:\t",tpp)
print("True negitive:\t",tnp)
print("False positive:\t",fpp)
print("False negitive:\t",fnp)
print("AUC:\t",auc)
print("Recall:\t",rec)
print("precision:\t",pre)
print("F1:\t",f1s)