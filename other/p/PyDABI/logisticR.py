#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:26:30 2019

@author: zahid
"""

import pandas as pd
from sklearn import datasets #Import scikit-learn dataset library
from sklearn.linear_model import LogisticRegression #Import Logistic regression Model
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import preprocessing # Import the preprocessing module


data = pd.read_csv('data/CleanDataROSE.csv')

feature_cols = ['Rainfallmm', 'X3pmTemperatureC',
       'X3pmRelativeHumidity', 'X3pmCloudAmountOktas', 'X3pmWindDirection',
       'X3pmWindSpeedkmh', 'X3pmMSLpressurehPa']
X = data[feature_cols] # Features
y = data['RainTomorrow'] # Target variable

le = preprocessing.LabelEncoder()
le.fit(X['X3pmWindDirection']) 
X['X3pmWindDirection'] = le.transform(X['X3pmWindDirection']) 

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test


#Create a Gaussian Classifier
clf=LogisticRegression()

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)


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

