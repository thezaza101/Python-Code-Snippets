import os
import numpy as np
import pandas as pd
import seaborn as sns
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import keras
from keras.wrappers.scikit_learn import KerasClassifier 
#use Sequential Keras models as part of your Scikit-Learn workflow via the wrappers

#Set working directory and load data
data_path =  os.path.abspath(os.path.join('other','p','iris','datasets'))

data  = pd.read_csv(data_path+'/iris.csv')
