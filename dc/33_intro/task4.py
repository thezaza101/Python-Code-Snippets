# Importing the os module 
import os as os
# Importing the pandas module
import pandas as pd
# Importing the matplotlib module
import matplotlib.pyplot as plt


# Define the path to the global temperature data
fullpath =  os.path.abspath(os.path.join('dc','33_intro','datasets', 'global_temperature.csv'))

# Reading in the global temperature data
global_temp = pd.read_csv(fullpath)

# Plotting global temperature in degrees celsius by year
plt.plot(global_temp['year'], global_temp['degrees_celsius'])

# Adding some nice labels 
plt.xlabel('...') 
plt.ylabel('...')

plt.show()