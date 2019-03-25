# 0.1 Download Data
'''
import wget

link_to_data = 'https://github.com/tulip-lab/sit742/raw/master/Assessment/2019/data/wine.json'
DataSet = wget.download(link_to_data, "/datasets")

link_to_data = 'https://github.com/tulip-lab/sit742/raw/master/Assessment/2019/data/stopwords.txt'

DataSet = wget.download(link_to_data,"/datasets")
'''
# 0.2 load data

import json
import pandas as pd
import matplotlib.pyplot as plt

file = 'datasets/wine.json'
data = pd.read_json(file)

print(data)