## https://projects.datacamp.com/projects/10
## Exploring 67 years of LEGO

## Task 1
# Nothing to do here

## Task 2
# Import modules
import pandas as pd
import os as os
import matplotlib.pyplot as plt

# Read colors data
coloursdataPath =  os.path.abspath(os.path.join('dc','10_lego_67_years','datasets', 'colors.csv'))
colors = pd.read_csv(coloursdataPath)

# Print the first few rows
print(colors.head())

## Task 3
# How many distinct colors are available?
num_colors = colors.shape[0]
print(num_colors)

## Task 4
# colors_summary: Distribution of colors based on transparency
colors_summary = colors.groupby('is_trans').count()
print(colors_summary)

## Task 5
# Read sets data as `sets`
setsdataPath =  os.path.abspath(os.path.join('dc','10_lego_67_years','datasets', 'sets.csv'))
sets = pd.read_csv(setsdataPath)
print(sets.head())

# Create a summary of average number of parts by year: `parts_by_year`
parts_by_year = sets.groupby('year').count()
print(parts_by_year)
# Plot trends in average number of parts by year
parts_by_year.plot()
# Show the plot
plt.show()

## Task 6
# themes_by_year: Number of themes shipped by year
themes_by_year = sets[['year', 'theme_id']].groupby('year', as_index = False).agg({"theme_id" : pd.Series.count})
print(themes_by_year.head())

## Task 7
# Nothing to do here
