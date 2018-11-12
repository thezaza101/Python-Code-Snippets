from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
import os

data_path =  os.path.abspath(os.path.join('intel','ml501','class1','datasets'))
#data_path = ['...........Intel-ML101-Class1\\Intel-ML101_Class1\\data\\']
print (data_path)

## Q1
filepath = data_path + "\\Iris_Data.csv"
print(filepath)
data = pd.read_csv(filepath)
print(data.head())

# Number of rows
print(data.shape[0])

# Column names
print(data.columns.tolist())

# Data types
print(data.dtypes)

## Q2
# The str method maps the following function to each entry as a string
data['species'] = data.species.str.replace('Iris-', '')
# alternatively
# data['species'] = data.species.apply(lambda r: r.replace('Iris-', ''))

print(data.head())

## Q3
##The number of each species presen
print(data.groupby('species').nunique())

## The mean, median, and quantiles and ranges (max-min) for each petal and sepal measurement.
desc = data.describe()
desc.loc["range"] = desc.loc['max'] - desc.loc['min']
print(desc)

## Q4
# The mean calculation
data.groupby('species').mean()

# The median calculation
data.groupby('species').median()

# applying multiple functions at once - 2 methods
data.groupby('species').agg(['mean', 'median'])  # passing a list of recognized strings
data.groupby('species').agg([np.mean, np.median])  # passing a list of explicit aggregation functions

# If certain fields need to be aggregated differently, we can do:

agg_dict = {field: ['mean', 'median'] for field in data.columns if field != 'species'}
agg_dict['petal_length'] = 'max'
pprint(agg_dict)
data.groupby('species').agg(agg_dict)

## Q5
# A simple scatter plot with Matplotlib
ax = plt.axes()

ax.scatter(data.sepal_length, data.sepal_width)

# Label the axes
ax.set(xlabel='Sepal Length (cm)',
       ylabel='Sepal Width (cm)',
       title='Sepal Length vs Width')

# show the plot
plt.show()

## Q6
ax = plt.hist(data['petal_length'])

plt.xlabel('petal length')
plt.ylabel('feq')

# show the plot
plt.show()

## Q7
sns.set_context('notebook')

# This uses the `.plot.hist` method
ax = data.plot.hist(bins=25, alpha=0.5)
ax.set_xlabel('Size (cm)')

# show the plot
plt.show()

# To create four separate plots, use Pandas `.hist` method
axList = data.hist(bins=25)

# Add some x- and y- labels to first column and last row
for ax in axList.flatten():
    if ax.is_last_row():
        ax.set_xlabel('Size (cm)')
        
    if ax.is_first_col():
        ax.set_ylabel('Frequency')

# show the plot
plt.show()

## Q8
axList = data.boxplot()

# show the plot
plt.show()

## Q9
# First we have to reshape the data so there is 
# only a single measurement in each column

plot_data = (data
             .set_index('species')
             .stack()
             .to_frame()
             .reset_index()
             .rename(columns={0:'size', 'level_1':'measurement'})
            )

plot_data.head()

# Now plot the dataframe from above using Seaborn

sns.set_style('white')
sns.set_context('notebook')
sns.set_palette('dark')

f = plt.figure(figsize=(6,4))
sns.boxplot(x='measurement', y='size', 
            hue='species', data=plot_data)

# show the plot
plt.show()

## Q10
xx = sns.pairplot(data, hue='species',size=3)
# show the plot
plt.show()
# xx.savefig("test.png")