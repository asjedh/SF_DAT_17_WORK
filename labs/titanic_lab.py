# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 19:40:00 2015

@author: asjedh
"""

import os
import pandas as pd
from sklearn.cross_validation import train_test_split

os.chdir("/Users/asjedh/Desktop/ga_data_science/SF_DAT_17_WORK/data")

titanic = pd.read_csv("titanic.csv")

titanic.info()

import seaborn as sns

sns.lmplot(y = "Survived", x = "Parch", data = titanic)


#split into train and test set

feature_cols = ["Pclass", "Parch"]
X = titanic[feature_cols]
y = titanic.Survived

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

logreg.fit(X_train, y_train)
zip(feature_cols, logreg.coef_[0])
feature_cols

test_pred = logreg.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, test_pred)

logreg.score(X_test, y_test)

from sklearn.metrics import confusion_matrix
titanic_cm = confusion_matrix(y_test, test_pred)
titanic_cm
#sensitivity - true negative rate
def calculate_sensitivity(cm):
    true_neg = cm[0][0]
    false_neg = cm[1][0]
    sensitivity = float(true_neg) / (true_neg + false_neg)
    return sensitivity
    
calculate_sensitivity(titanic_cm)

#specificity - true positive rate
def calculate_specificity(cm):
    true_pos = cm[1][1]
    false_pos = cm[0][1]
    specificity = float(true_pos) / (true_pos + false_pos)
    return specificity
    
calculate_specificity(titanic_cm)



