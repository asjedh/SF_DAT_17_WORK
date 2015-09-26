'''
Move this code into your OWN SF_DAT_15_WORK repo

Please complete each question using 100% python code

If you have any questions, ask a peer or one of the instructors!

When you are done, add, commit, and push up to your repo

This is due 7/1/2015
'''


import pandas as pd
# pd.set_option('max_colwidth', 50)
# set this if you need to

killings = pd.read_csv('data/police-killings.csv')
killings.head()

# 1. Make the following changed to column names:
# lawenforcementagency -> agency
# raceethnicity        -> race
killings.rename(columns = {"lawenforcementagency": "agency", "raceethnicity": "race" }, inplace = True)    
killings.columns
    
# 2. Show the count of missing values in each column
killings.isnull().sum()

# 3. replace each null value in the dataframe with the string "Unknown"
killings.fillna("Unknown")
# 4. How many killings were there so far in 2015?
killings[killings.year == 2015].shape[0] # 467

# 5. Of all killings, how many were male and how many female?
killings[killings.gender == "Male"].shape[0] # male: 445
killings[killings.gender == "Female"].shape[0] # female: 22

# 6. How many killings were of unarmed people?
killings.armed.value_counts().No #102

# 7. What percentage of all killings were unarmed?
(killings.armed.value_counts().No / float(len(killings))) * 100 # 21.84%

# 8. What are the 5 states with the most killings?
killings.state.value_counts().head(5)

# 9. Show a value counts of deaths for each race
killings.race.value_counts()

# 10. Display a histogram of ages of all killings
killings.age.hist()

# 11. Show 6 histograms of ages by race
killings.age.hist(by = killings.race)

# 12. What is the average age of death by race?
killings.info()
killings.groupby("race").age.mean()
# 13. Show a bar chart with counts of deaths every month
killings.groupby("month").count().state.plot(kind = "bar")

###################
### Less Morbid ###
###################

majors = pd.read_csv('data/college-majors.csv')
majors.head()

# 1. Delete the columns (employed_full_time_year_round, major_code)
majors.drop(["Employed_full_time_year_round", "Major_code"], axis = 1, inplace = True)
majors.info()

# 2. Show the count of missing values in each column
majors.isnull().sum()

# 3. What are the top 10 highest paying majors?
#Under which definition...?

majors.sort_index(by="Median", ascending = False).head(10).Major

# 4. Plot the data from the last question in a bar chart, include proper title, and labels!
top_majors_barplot = majors.sort_index(by="Median", ascending = False).head(10).plot(kind = "bar", x = "Major", y = "Median", legend = False, title = "Median Income by Major")
top_majors_barplot.set_xlabel("Major Name")
top_majors_barplot.set_ylabel("Median Income")

# 5. What is the average median salary for each major category?
majors.groupby("Major_category").Median.mean()
# 6. Show only the top 5 paying major categories
majors.groupby("Major_category").Median.mean().order(ascending = False).head(5)

# 7. Plot a histogram of the distribution of median salaries
majors.Median.hist()

# 8. Plot a histogram of the distribution of median salaries by major category
majors.groupby("Major_category").Median.mean().hist()

# 9. What are the top 10 most UNemployed majors?
majors.sort_index(by = "Unemployed", ascending = False).head(10)

# What are the unemployment rates?
majors.sort_index(by = "Unemployed", ascending = False).head(10)[["Major", "Unemployment_rate"]]

# 10. What are the top 10 most UNemployed majors CATEGORIES? Use the mean for each category
majors.groupby("Major_category").Unemployed.mean().order(ascending = False).head(10)

# What are the unemployment rates?
majors.groupby("Major_category")[["Major", "Unemployed", "Unemployment_rate"]].mean().sort_index(by = "Unemployed", ascending = False).head(10)

# 11. the total and employed column refer to the people that were surveyed.
# Create a new column showing the emlpoyment rate of the people surveyed for each major
# call it "sample_employment_rate"
# Example the first row has total: 128148 and employed: 90245. it's 
# sample_employment_rate should be 90245.0 / 128148.0 = .7042
majors.info()
employment_rates = majors.Employed / majors.Total
employment_rates

majors["sample_employment_rate"] = employment_rates

# 12. Create a "sample_unemployment_rate" colun
# this column should be 1 - "sample_employment_rate"
majors["sample_unemployment_rate"] = 1 - majors["sample_employment_rate"]

