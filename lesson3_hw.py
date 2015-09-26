# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 06:58:40 2015

@author: asjedh
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('max_columns', 50)
%matplotlib inline

s = pd.Series([7, "Heisenberg",
                3.14, -3443, "Happy eating!"])
s

s = pd.Series([7, "Heisenberg",
                3.14, -3443, "Happy eating!"],
                index = ["a", "b", "c", "d", "e"])
s


d = {'Chicago': 1000, "New York": 1300,
     'Portland': 900, "SF": 1100, 'Austin': 450,
     'Boston': None}

cities = pd.Series(d)
cities

cities['Chicago']
cities[['Chicago', 'Boston']]

cities[cities < 1000]

print "Old value:", cities["Chicago"]
cities["Chicago"] = 1400
print "New value:", cities["Chicago"]

cities[cities < 1000] = 750
print cities[cities<1000]

#Why do these commands work this way? Why don't they print the value?
print "Seattle" in cities
print "SF" in cities

#math operations
cities / 3
np.square(cities)



