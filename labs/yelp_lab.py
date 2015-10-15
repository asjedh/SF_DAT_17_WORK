# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 20:55:06 2015

@author: asjedh
"""
import pandas as pd
import os
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import numpy as np

os.chdir("/Users/asjedh/Desktop/ga_data_science/SF_DAT_17_WORK/data")
yelp = pd.read_csv("yelp.csv")

yelp.info()
yelp.head()
# scatter plots in yelp data
yelp.plot(kind = "scatter", x = "cool", y = "stars")

sns.pairplot(yelp, x_vars = "cool", y_vars = "stars", size = 6)
sns.pairplot(yelp, x_vars = ["cool", "funny", "useful"], y_vars = "stars", size = 4)
sns.pairplot(yelp)

#check correlations
yelp.corr()


#build linear model
features_cols = ["cool", "funny", "useful"]
X = yelp[features_cols]
y = yelp.stars

yelp_linreg = LinearRegression()
yelp_linreg.fit(X, y)

yelp_linreg.intercept_
yelp_linreg.coef_

zip(features_cols, yelp_linreg.coef_)

#stat model
yelp_smf_linreg = smf.ols(formula = "stars ~ cool + funny + useful", data = yelp).fit()
yelp_smf_linreg.params
yelp_smf_linreg.pvalues
yelp_smf_linreg.conf_int()
yelp_smf_linreg.rsquared

# split, train, test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)
len(X_train)
len(X_test)

yelp_linreg.fit(X_train, y_train)
zip(features_cols, yelp_linreg.coef_)


y_pred = yelp_linreg.predict(X_test)

y_pred.shape
y_test.shape

metrics.mean_squared_error(y_test, y_pred)
np.sqrt(metrics.mean_squared_error(y_test, y_pred))


# define TTS + RMSE function
def train_test_rmse(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    
    y_test_pred = linreg.predict(X_test)
    
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
    return rmse

train_test_rmse(X, y) #best model


features_cols = ["cool", "funny"]
X2 = yelp[features_cols]
train_test_rmse(X2, y)

features_cols = ["cool"]
X3 = yelp[features_cols]
train_test_rmse(X3, y)

features_cols = ["cool", "useful"]
X4 = yelp[features_cols]
train_test_rmse(X4, y)

features_cols = ["useful", "funny"]
X5 = yelp[features_cols]
train_test_rmse(X5, y)

features_cols = ["useful"]
X6 = yelp[features_cols]
train_test_rmse(X6, y)

features_cols = ["funny"]
X7 = yelp[features_cols]
train_test_rmse(X7, y)

# BONUSES

#Bonus: Think of some new features you could create from the existing data that might be predictive of the response. (This is called "feature engineering".) Figure out how to create those features in Pandas, add them to your model, and see if the RMSE improves.
#Bonus: Compare your best RMSE on testing set with the RMSE for the "null model", which is the model that ignores all features and simply predicts the mean rating in the training set for all observations in the testing set.
#Bonus: Instead of treating this as a regression problem, treat it as a classification problem and see what testing accuracy you can achieve with KNN.
#Bonus: Figure out how to use linear regression for classification, and compare its classification accuracy to KNN.

yelp.info()
yelp.head()
yelp.type.describe()
len(yelp.text[0])
len(yelp.text[1])


yelp["text_length"] = yelp.text.apply(lambda x: len(x))
yelp.text_length

features_cols = ["useful", "funny", "cool", "text_length"]
X = yelp[features_cols]
yelp_linreg.fit(X, y)
zip(features_cols, yelp_linreg.coef_)

yelp_smf_linreg = smf.ols(formula = "stars ~ funny + cool + useful + text_length", data = yelp).fit()
yelp_smf_linreg.pvalues
yelp_smf_linreg.params

train_test_rmse(X, y)

null_prediction = [yelp.stars.mean()] * 10000

np.sqrt(metrics.mean_squared_error(yelp.stars, null_prediction))

#treat it as a classification problem
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5)

features_cols = ["useful", "funny", "cool", "text_length"]
knn.fit(X,y)
knn.score(X, y)










