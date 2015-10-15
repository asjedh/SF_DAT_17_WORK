# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 19:04:59 2015

@author: asjedh
"""

import os
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression

os.chdir("/Users/asjedh/Desktop/ga_data_science/SF_DAT_17_WORK/data")

stocks = pd.read_csv("ZYX_prices.csv")

stocks.info()

stocks.ZYX30minSentiment.describe()

stocks.corr()

sns.pairplot(data = stocks, y_vars=["60fret"], x_vars = ["ZYX1MinSentiment", "ZYX1MinTweets" ], aspect = 2)

sns.pairplot(data = stocks, y_vars=["60fret"], x_vars = ["ZYX1MinSentiment", "ZYX1MinTweets" ], aspect = 2)


logreg = LogisticRegression()

stocks["60fret"] >= 0
stocks["60Direction"] = stocks["60fret"] >= 0
sum(stocks["60Direction"])

y = stocks["60Direction"]
y.describe()

feature_cols = ["ZYX1MinSentiment", "ZYX5minSentiment"]
X = stocks[feature_cols]
X

X
logreg.fit(X, y)
logreg.score(X, y)
stocks.info()

feature_cols = ["ZYX1MinSentiment"]
X = stocks[feature_cols]

logreg.fit(X, y)
logreg.score(X, y)

feature_cols = ["ZYX5MinSentiment"]
X = stocks[feature_cols]
X

logreg.fit(X, y)
logreg.score(X, y)

stocks.info()
feature_cols = ["ZYX20MinSentiment"]
X = stocks[feature_cols]
X

logreg.fit(X, y)
logreg.score(X, y)

stocks.info()
feature_cols = ["ZYX30MinSentiment"]
X = stocks[feature_cols]
X

logreg.fit(X, y)
logreg.score(X, y)

stocks.info()
feature_cols = ["ZYX1MinSentiment", "ZYX5minSentiment", "ZYX10minSentiment", "ZYX20minSentiment", "ZYX30minSentiment", "ZYX60minSentiment"]
X = stocks[feature_cols]
X

logreg.fit(X, y)
logreg.score(X, y)

feature_cols = ["ZYX1MinSentiment", "ZYX5minSentiment", "ZYX10minSentiment", "ZYX20minSentiment", "ZYX30minSentiment", "ZYX60minSentiment"]
X = stocks[feature_cols]
X

logreg.fit(X, y)
logreg.score(X, y)

stocks.ZYX1MinTweets.describe()

stocks.ZYX1MinTweets[stocks.ZYX1MinTweets == 0] = 1
stocks.ZYX1MinTweets.describe()

stocks["one_min_avg_sent"] = stocks.ZYX1MinSentiment / stocks.ZYX1MinTweets

stocks.OneMinuteAverageSentiment.describe()


feature_cols = ["ZYX1MinSentiment"]
X = stocks[feature_cols]
X

logreg.fit(X, y)
logreg.score(X, y)

feature_cols = ["OneMinuteAverageSentiment"]
X = stocks[feature_cols]
X

logreg.fit(X, y)
logreg.score(X, y)

stocks.info()
feature_cols = ["ZYX20minPriceChange"]
X = stocks[feature_cols]
X

logreg.fit(X, y)
logreg.score(X, y)

#Change 0 Tweets to 1
stocks.ZYX5minTweets[stocks.ZYX5minTweets == 0] = 1
stocks.ZYX10minTweets[stocks.ZYX10minTweets == 0] = 1
stocks.ZYX20minTweets[stocks.ZYX20minTweets == 0] = 1
stocks.ZYX30minTweets[stocks.ZYX30minTweets == 0] = 1
stocks.ZYX60minTweets[stocks.ZYX60minTweets == 0] = 1

#calculate average sentiments
stocks["five_min_avg_sent"] = stocks.ZYX5minSentiment / stocks.ZYX5minTweets
stocks.five_min_avg_sent.describe()

stocks["ten_min_avg_sent"] = stocks.ZYX10minSentiment / stocks.ZYX10minTweets

stocks["twenty_min_avg_sent"] = stocks.ZYX20minSentiment / stocks.ZYX20minTweets

stocks["thirty_min_avg_sent"] = stocks.ZYX30minSentiment / stocks.ZYX30minTweets

stocks["sixty_min_avg_sent"] = stocks.ZYX60minSentiment / stocks.ZYX60minTweets

# try regression with averages
stocks.info()
feature_cols = ["ten_min_avg_sent"]
X = stocks[feature_cols]
X

logreg.fit(X, y)
logreg.score(X, y)

feature_cols = ["twenty_min_avg_sent"]
X = stocks[feature_cols]
X

logreg.fit(X, y)
logreg.score(X, y)

stocks.info()

feature_cols = ["thirty_min_avg_sent", "ten_min_avg_sent", "twenty_min_avg_sent", "five_min_avg_sent", "sixty_min_avg_sent"]
X = stocks[feature_cols]
X

logreg.fit(X, y)
logreg.score(X, y)


feature_cols = ["thirty_min_avg_sent", "ten_min_avg_sent", "twenty_min_avg_sent", "five_min_avg_sent", "sixty_min_avg_sent"]
X = stocks[feature_cols]
X

logreg.fit(X, y)
logreg.score(X, y)
zip(feature_cols, logreg.coef_[0])

sum(stocks["60Direction"]) / float(len(stocks["60Direction"]))

##### 1)  LOOK AT CORRELATIONS
stocks.info()

stocks[["60Direction", "ZYX60minSentiment", "ZYX30minSentiment", "ZYX20minSentiment", "ZYX10minSentiment", "ZYX5minSentiment", "ZYX1MinSentiment"]].corr()

stocks[["60Direction", "ZYX60minTweets", "ZYX30minTweets", "ZYX20minTweets", "ZYX10minTweets", "ZYX5minTweets", "ZYX1MinTweets"]].corr()


stocks[["60Direction", "ZYX60minPriceChange", "ZYX30minPriceChange", "ZYX20minPriceChange", "ZYX10minPriceChange", "ZYX5minPriceChange", "ZYX1minPriceChange"]].corr()


stocks[["60Direction", "one_min_avg_sent", "five_min_avg_sent", "ten_min_avg_sent", "twenty_min_avg_sent", "thirty_min_avg_sent", "sixty_min_avg_sent"]].corr()

# fit new models
feature_cols = ["sixty_min_avg_sent", "ZYX60minSentiment"]
X = stocks[feature_cols]
X

logreg.fit(X, y)
logreg.score(X, y)

stocks.info()

feature_cols = ["sixty_min_avg_sent"]
X = stocks[feature_cols]
X

logreg.fit(X, y)
logreg.score(X, y)

feature_cols = ["ZYX60minSentiment"]
X = stocks[feature_cols]
X

logreg.fit(X, y)
logreg.score(X, y)



y_pred = logreg.predict(X)

import numpy as np
logodds = logreg.intercept_
odds = np.exp(logodds)
prob = odds / (1 + odds)
prob

import matplotlib as plt

plt.scatter(stocks["60Direction"], X)

from sklearn import metrics

metrics.confusion_matrix(y, y_pred)

# 2) TRAIN TEST SPLIT
from sklearn import cross_validation

#both variables
feature_cols = ["sixty_min_avg_sent", "ZYX60minSentiment"]
X = stocks[feature_cols]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, random_state = 0)

logreg.fit(X_train, y_train)

logreg.score(X_test, y_test)


#60 min avg only
feature_cols = ["sixty_min_avg_sent"]
X = stocks[feature_cols]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, random_state = 0)

logreg.fit(X_train, y_train)

logreg.score(X_test, y_test)

#60 min sentiment only
feature_cols = ["ZYX60minSentiment"]
X = stocks[feature_cols]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, random_state = 0)

logreg.fit(X_train, y_train)

logreg.score(X_test, y_test)

cm = metrics.confusion_matrix(y_test, y_pred)

calculate_sensitivity(cm)
calculate_specificity(cm)



#specificity - true negative rate
def calculate_sensitivity(cm):
    true_neg = cm[0][0]
    false_neg = cm[1][0]
    sensitivity = float(true_neg) / (true_neg + false_neg)
    return sensitivity
    
calculate_sensitivity(cm)

#specificity - true positive rate
def calculate_specificity(cm):
    true_pos = cm[1][1]
    false_pos = cm[0][1]
    specificity = float(true_pos) / (true_pos + false_pos)
    return specificity
    
calculate_specificity(cm)


stocks.time = pd.to_datetime(stocks.time, format = "%m/%d/%y %H:%M")
stocks.time[0].hour < 12

stocks["morning"] = stocks.time.apply(lambda x: x.hour < 12)
sum(stocks.morning)

stocks["afternoon"] = stocks.time.apply(lambda x: (x.hour > 12) & (x.hour < 15))
sum(stocks.afternoon)

stocks["evening"] = stocks.time.apply(lambda x: x.hour > 15)
sum(stocks.evening)

# 3) CALCULATE WITH TIME VARIABLES
feature_cols = ["ZYX60minSentiment", "morning", "afternoon", "evening"]
X = stocks[feature_cols]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, random_state = 0)

logreg.fit(X_train, y_train)

logreg.score(X_test, y_test)

y_pred = logreg.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_pred)
calculate_sensitivity(cm)
calculate_specificity(cm)

#calculate with average time sentiment as well
feature_cols = ["ZYX60minSentiment", "morning", "afternoon", "evening", "sixty_min_avg_sent"]
X = stocks[feature_cols]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, random_state = 0)

logreg.fit(X_train, y_train)

logreg.score(X_test, y_test)

y_pred = logreg.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_pred)
calculate_sensitivity(cm)
calculate_specificity(cm)




### KNN Model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
knn.score(X_test, y_test)

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
knn.score(X_test, y_test)

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
knn.score(X_test, y_test)

knn = KNeighborsClassifier(n_neighbors = 7)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
knn.score(X_test, y_test)




