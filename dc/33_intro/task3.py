# Importing the os module 
import os as os
# Importing the pandas module
import pandas as pd

# Define the path to the global temperature data
fullpath =  os.path.abspath(os.path.join('dc','33_intro','datasets', 'global_temperature.csv'))

# Reading in the global temperature data
global_temp = pd.read_csv(fullpath)

print(global_temp.head())