# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 19:21:12 2015

@author: asjedh
"""
# imports
import pandas as pd
import os as os
import time
# Set WD

os.chdir("/Users/asjedh/Desktop/ga_data_science/SF_DAT_17_WORK/project")

#read data
disasters = pd.read_csv("1999-2013_National_Natural_Disaster_Inventory.csv")

# format colums
new_cols = [col.lower().replace(" ", "_").replace(":", "").replace("\xc2\xa0", "").replace("$", "") for col in disasters.columns]
new_cols
disasters.columns = new_cols
disasters.columns

# inspect data
disasters.info()
disasters.county_centriod
disasters.losses_usd.mean()
disasters.losses_local.mean()
disasters.affected
disasters.event.value_counts()
disasters.date

# change date to time object

# remove incorrect data
disasters = disasters[disasters.date.notnull()]
new_times = [time.strptime(date, "%m/%d/%Y %H:%M:%S %p") for date in disasters.date]
disasters.date = new_times

# create year and months column
disasters["year"] = disasters.date.apply(lambda time: time.tm_year)
disasters.year

disasters["month"] = disasters.date.apply(lambda time: time.tm_mon)
disasters.month

# look at data by time
disasters.year.value_counts()
#lots of cases in 2009. Looking online, it seems there was a drought
disasters.month.hist(by = disasters.year)

# many cattle died in the drought. Let's check it out
disasters.info()
disasters.affected

disasters.groupby("event").county.value_counts()



