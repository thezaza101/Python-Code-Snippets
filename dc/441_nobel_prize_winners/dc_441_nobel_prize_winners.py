# https://projects.datacamp.com/projects/441
# A Visual History of Nobel Prize Winners

## Task 1
# Loading in required libraries
import pandas as pd
import seaborn as sns
import numpy as np
import os as os

# Reading in the Nobel Prize data
fullnobelearly =  os.path.abspath(os.path.join('dc','441_nobel_prize_winners','datasets', 'nobel.csv'))
nobel = pd.read_csv(fullnobelearly)

# Taking a look at the first several winners
print(nobel.head(5))

## Task 2
# Display the number of (possibly shared) Nobel Prizes handed
# out between 1901 and 2016
print(len(nobel))

# Display the number of prizes won by male and female recipients.
print(nobel["sex"].value_counts())

# Display the number of prizes won by the top 10 nationalities.
winnerByNational = nobel["birth_country"].value_counts()

print(winnerByNational.head(10))

## Task 3
# Calculating the proportion of USA born winners per decade
nobel['usa_born_winner'] = nobel['birth_country'] == 'United States of America'
nobel['decade'] = (np.floor(nobel["year"] / 10) * 10).astype(int)
prop_usa_winners = nobel.groupby("decade", as_index=False)["usa_born_winner"].mean()

# Display the proportions of USA born winners per decade
print(prop_usa_winners)

## Task 4
# Setting the plotting theme
sns.set()
# and setting the size of all plots.
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [11, 7]

# Plotting USA born winners 
ax = sns.lineplot(x=nobel["decade"], y=nobel["usa_born_winner"])

# Adding %-formatting to the y-axis
from matplotlib.ticker import PercentFormatter
ax.yaxis.set_major_formatter(PercentFormatter())

# Show the plot
# plt.show()

## Task 5
# Calculating the proportion of female laureates per decade
nobel['female_winner'] = nobel['sex'] == 'Female'
prop_female_winners = nobel.groupby(["decade","category"], as_index=False)["female_winner"].mean()

# Plotting USA born winners with % winners on the y-axis
ax = sns.lineplot(x="decade", y="female_winner", hue="category", data=prop_female_winners)
ax.yaxis.set_major_formatter(PercentFormatter())

# Show the plot
# plt.show()

## Task 6
# Picking out the first woman to win a Nobel Prize
print(nobel[nobel["sex"] == "Female"].nsmallest(1,"year"))

## Task 7
# Selecting the laureates that have received 2 or more prizes.
print(nobel.groupby("full_name", as_index=False).filter(lambda group: len(group) >= 2))

## Task 8
# Converting birth_date from String to datetime
nobel['birth_date'] = pd.to_datetime(nobel["birth_date"])

# Calculating the age of Nobel Prize winners
nobel['age'] = nobel["year"] - nobel["birth_date"].dt.year

# Plotting the age of Nobel Prize winners
sns.lmplot(x="year", y="age", data=nobel, lowess=True, aspect=2, line_kws={"color" : "black"})

# Show the plot
# plt.show()

## Task 9
# Same plot as above, but separate plots for each type of Nobel Prize
sns.lmplot(x="year", y="age", data=nobel, row="category", lowess=True, aspect=2, line_kws={"color" : "black"})

# Show the plot
# plt.show()

## Task 10
# The oldest winner of a Nobel Prize as of 2016
print(nobel.nlargest(1,"age"))

# The youngest winner of a Nobel Prize as of 2016
print(nobel.nsmallest(1,"age"))

## Task 11
# The name of the youngest winner of the Nobel Prize as of 2016
youngest_winner = 'Malala Yousafzai'