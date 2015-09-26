import pandas as pd


'''
Part 1: UFO

'''

ufo = pd.read_csv('https://raw.githubusercontent.com/sinanuozdemir/SF_DAT_17/master/data/ufo.csv')   # can also read csvs directly from the web!



# 1. change the column names so that each name has no spaces
#           and all lower case
new_cols = [col.lower().replace(" ", "").replace("reported", "") for col in ufo.columns]
new_cols[2] = "shapes"

ufo.columns = new_cols
ufo


# 2. Show a bar chart of all shapes reported

ufo.shapes.value_counts().plot(kind = "bar")

# 3. Show a dataframe that only displays the reportings from Utah
ufo.describe()
ufo[ufo.state == "UT"]

# 4. Show a dataframe that only displays the reportings from Texas

ufo[ufo.state == "TX"]

# 5. Show a dataframe that only displays the reportings from Utah OR Texas
ufo[(ufo.state == "TX") | (ufo.state == "UT")]

# 6. Which shape is reported most often?
ufo.shapes.value_counts().head(1)


# 7. Plot number of sightings per day in 2014 (days should be in order!)
ufo.describe()

ufo["date"] = ufo.time.apply(lambda x: x.split(" ")[0].replace(" ", ""))

dates_as_datetimes = ufo.date.apply(lambda x: pd.to_datetime(x))
dates_as_datetimes

ufo["date"] = dates_as_datetimes

ufo.date[0].year

ufo["year"] = ufo.date.apply(lambda x: x.year)
ufo.year
ufo["year"] = ufo.date.apply(lambda x: x.year)

ufo[ufo.year == 2014].date.value_counts().sort_index().plot(kind = 'bar')

'''
Part 2: IRIS

'''


iris = pd.read_csv('https://raw.githubusercontent.com/sinanuozdemir/SF_DAT_17/master/data/iris.csv')   # can also read csvs directly from the web!

# 1. Show the mean petal length by flower species
iris.describe()
iris.info()

iris.groupby("species").petal_length.mean()

# 2. Show the mean sepal width by flower species
iris.groupby("species").sepal_width.mean()

# 3. Use the groupby to show both #1 and #2 in one dataframe
iris[["sepal_width", "petal_length"]]
iris.groupby("species").mean()[["sepal_width", "petal_length"]]

# 4. Create a scatter plot plotting petal length against petal width

iris[["sepal_width", "petal_length"]].plot(kind = "scatter",
x = "sepal_width", y = "petal_length")

# 5. Show flowers with sepal length over 5 and petal length under 1.5
iris[(iris.sepal_length > 5) & (iris.petal_length < 1.5)]
iris[(iris.sepal_width > 3) & (iris.petal_length < 2.5)]


# 6. Show setosa flowers with petal width of exactly 0.2
iris[(iris.species == "Iris-setosa") & (iris.petal_width == 0.2)]


# 7. Write a function to predict the species for each observation
iris.groupby(iris.species).describe()

iris.sort_index(by='sepal_length') # good
iris.sort_index(by='sepal_width') # not good
iris.sort_index(by='petal_length') # better
iris.sort_index(by='petal_width') # better

iris.petal_length.hist(by = iris.species, sharex = True)
iris.petal_width.hist(by = iris.species, sharex = True)



def classify_iris(data):
    if data[2] < 2.5:
        return 'Iris-setosa'
    elif (data[2] < 5) & (data[3] < 1.75):
        return 'Iris-versicolor'
    else:
        return 'Iris-virginica'



# example use: 
# classify_iris([0,3,2.1,3.2]) == 'Iris-virginica'
# assume the order is the same as the dataframe, so:
# [sepal_length', 'sepal_width', 'petal_length', 'petal_width']


# make predictions and store as preds
preds = iris.drop('species', axis=1).apply(classify_iris, axis = 1)


preds


# test your function: compute accuracy of your prediction
(preds == iris['species']).sum() / float(iris.shape[0])


'''
Part 3: FIFA GOALS

'''

goals = pd.read_csv('https://raw.githubusercontent.com/sinanuozdemir/SF_DAT_17/master/data/fifa_goals.csv')
# removing '+' from minute and turning them into ints
goals.minute = goals.minute.apply(lambda x: int(x.replace('+','')))


goals.head()


# 1. Show goals scored in the first 5 minutes of a game


# 2. Show goals scored after the regulation 90 minutes is over


# 3. Show the top scoring players


# 4. Show a histogram of minutes with 20 bins

# 5. Show a histogram of the number of goals scored by players

