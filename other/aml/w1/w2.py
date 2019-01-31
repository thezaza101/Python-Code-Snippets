import os
data_path =  os.path.abspath(os.path.join('other','aml','w1','datasets'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Read in the data
tr_path = data_path+'/train.csv'
test_path = data_path+'/test.csv'

data = pd.read_csv(tr_path)  

### The .head() function shows the first few lines of data for perspecitve
print(data.head())

### GRADED
### Which column has the most "null" values? assign name as string to ans1.
### ### CAPITALIZATION/SPELLING MATTERS e.g. 'Street' != 'street'
### How many nulls are in that column? assign number as int to ans2
### YOUR ANSWER BELOW

null_columns=data.columns[data.isnull().any()]
print(data[null_columns].isnull().sum())

### Practice subsetting a DataFrame below.
### Create a DataFrame only containing the "Street" and "Alley" columns from 
### the `data` DataFrame.### Practice subsetting a DataFrame below.
### Create a DataFrame only containing the "Street" and "Alley" columns from 
### the `data` DataFrame.

ans1 = pd.DataFrame(data[['Street','Alley']])
print(ans1.head())

def standardize(num_list):
    avr = np.mean(num_list)
    std = np.std(num_list)
    output = []
    for num in num_list:
        output.append((num-avr)/std)
    """
    Standardize the given list of numbers
    
    Positional arguments:
        num_list -- a list of numbers
    
    Example:
        num_list = [1,2,3,3,4,4,5,5,5,5,5]
        nl_std = standardize(num_list)
        print(np.round(nl_std,2))
        #--> np.array([-2.11, -1.36, -0.61, -0.61,  
                0.14,  0.14,  0.88,  0.88,  0.88,
                0.88,  0.88])
    """
    
    
    return output

#print(standardize(data['SalePrice']))


def preprocess_for_regularization(data, y_column_name, x_column_names):
    output_df = pd.DataFrame()
    for col in x_column_names:
        output_df[col] = standardize(data[col])
    
    ydata = data[y_column_name]
    ymean = np.mean(ydata)
    ydata = ydata - ymean
    output_df[y_column_name] = ydata
    
    
    
    """
    Perform mean subtraction and dimension standardization on data
        
    Positional argument:
        data -- a pandas dataframe of the data to pre-process
        y_column_name -- the name (string) of the column that contains
            the target of the training data.
        x_column_names -- a *list* of the names of columns that contain the
            observations to be standardized
        
    Returns:
        Return a DataFrame consisting only of the columns included
        in `y_column_name` and `x_column_names`.
        Where the y_column has been mean-centered, and the
        x_columns have been mean-centered/standardized.
        
        
    Example:
        data = pd.read_csv(tr_path).head()
        prepro_data = preprocess_for_regularization(data,'SalePrice', ['GrLivArea','YearBuilt'])
        
        print(prepro_data) #-->
                   GrLivArea  YearBuilt  SalePrice
                0  -0.082772   0.716753     7800.0
                1  -1.590161  -0.089594   -19200.0
                2   0.172946   0.657024    22800.0
                3  -0.059219  -1.911342   -60700.0
                4   1.559205   0.627159    49300.0
    """
    return output_df

print(preprocess_for_regularization(data,'SalePrice', ['GrLivArea','YearBuilt']))

def ridge_regression_weights(input_x, output_y, lambda_param):
    X = input_x
    Y = output_y
    output_df = pd.DataFrame()
    # Step 1
    if (len(X) < X.shape[1]):
        X = X.transpose()
    # Step 2
    X = np.concatenate((np.ones((X.shape[0],1), dtype=int), X), axis=1)
    # Step 3
    Xt = X.transpose()
    lI = lambda_param*np.identity(len(Xt))
    mmInv = np.linalg.inv(np.dot(Xt, X)+lI)
    output_df = np.dot(np.dot(mmInv, Xt), Y)
    weights = np.array(output_df)
    return weights

training_y = np.array([208500, 181500, 223500, 
                                140000, 250000, 143000, 
                                307000, 200000, 129900, 
                                118000])
                                
training_x = np.array([[1710, 1262, 1786, 
                        1717, 2198, 1362, 
                        1694, 2090, 1774, 
                        1077], 
                        [2003, 1976, 2001, 
                        1915, 2000, 1993, 
                        2004, 1973, 1931, 
                        1939]])
lambda_param = 10

rrw = ridge_regression_weights(training_x, training_y, lambda_param)

print(rrw) #--> np.array([-576.67947107,   77.45913349,   31.50189177])
print(rrw[2]) #--> 31.50189177