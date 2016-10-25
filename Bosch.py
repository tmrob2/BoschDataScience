import numpy as np
import pandas as pd
import collections 

#do an initial read of the data to determine the column count
td = pd.read_csv("C:/Users/Thomas/Documents/Kaggle/Bosch/train_categorical.csv", nrows = 10)

data = pd.read_csv("C:/Users/Thomas/Documents/Kaggle/Bosch/train_categorical.csv",
                  chunksize = 100000, dtype = str, usecols=list(range(1,2141)))

uniques = collections.defaultdict(set)

#load the columns into a dictionary which has the column name and a list of the column entries that are not null
for chunk in data:
    for col in chunk:
        uniques[col] = uniques[col].union(chunk[col][chunk[col].notnull()].unique())
#print a list fo all of the empty columns
empty = 0
for key in uniques:
    if len(uniques[key])==0:
        print(key)
        empty = empty + 1 

#print a list of all of the columns only containing one value
single = 0
singles = []
for key in uniques:
    if len(uniques[key]) == 1:
        print(key, uniques[key])
        singles.append(uniques[key])
        single = single + 1

#print a list of the columns containing multiple values
multi = 0
multiples = []
for key in uniques:
    if len(uniques[key]) > 1:
        print(key, uniques[key])
        multiples.append(uniques[key])
        multi = multi + 1
