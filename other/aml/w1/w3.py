import os
data_path =  os.path.abspath(os.path.join('other','aml','w1','datasets'))

### This cell imports the necessary modules and sets a few plotting parameters for display

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)

### GRADED
### Code a function called `calc_posterior`

### ACCEPT three inputs
### Two floats: the likelihood and the prior
### One list of tuples, where each tuple has two values corresponding to:
### ### ( P(Bn) , P(A|Bn) )
### ### ### Assume the list of tuples accounts for all potential values of B
### ### ### And that those values of B are all mutually exclusive.
### The list of tuples allows for the calculation of normalization constant.

### RETURN a float corresponding to the posterior probability

### YOUR ANSWER BELOW

def calc_posterior(likelihood, prior, norm_list):


    """
    Calculate the posterior probability given likelihood,
    prior, and normalization
    
    Positional Arguments:
        likelihood -- float, between 0 and 1
        prior -- float, between 0 and 1
        norm_list -- list of tuples, each tuple has two values
            the first value corresponding to the probability of a value of "b"
            the second value corresponding to the probability of 
                a value of "a" given that value of "b"
    Example:
        likelihood = .8
        prior = .3
        norm_list = [(.25 , .9), (.5, .5), (.25,.2)]
        print(calc_posterior(likelihood, prior, norm_list))
        # --> 0.45714285714285713
    """
    Pa = 0
    
    for t in norm_list:
        x = t[0] * t[1]
        Pa+=x

    return (likelihood*prior)/Pa

likelihood = .8
prior = .3
norm_list = [(.25 , .9), (.5, .5), (.25,.2)]
print(calc_posterior(likelihood, prior, norm_list))

def euclid_dist(p1, p2):
    
    """
    Calculate the Euclidian Distance between two points
    
    Positional Arguments:
        p1 -- A tuple of n numbers
        p2 -- A tuple of n numbers
    
    Example:
        p1 = (5,5)
        p2 = (0,0)
        p3 = (5,6,7,8,9,10)
        p4 = (1,2,3,4,5,6)
        print(euclid_dist(p1,p2)) #--> 7.0710678118654755
        print(euclid_dist(p3,p4)) #--> 9.797958971132712
    """
    
    return float(np.linalg.norm(np.array(p1)-np.array(p2)))

p1 = (5,5)
p2 = (0,0)
p3 = (5,6,7,8,9,10)
p4 = (1,2,3,4,5,6)
print(euclid_dist(p1,p2))
print(euclid_dist(p3,p4))

### GRADED
### Build a function called "x_preprocess"
### ACCEPT one input, a numpy array
### ### Array may be one or two dimensions

### If input is two dimensional, make sure there are more rows than columns
### ### Then prepend a column of ones for intercept term
### If input is one-dimensional, prepend a one

### RETURN a numpy array, prepared as described above,
### which is now ready for matrix multiplication with regression weights

def x_preprocess(input_x):
    if (input_x.ndim==2):
        if (len(input_x) < input_x.shape[1]):
            input_x = input_x.transpose()
    if (input_x.ndim==2):
        input_x = np.concatenate((np.ones((input_x.shape[0],1), dtype=int), input_x), axis=1)

    if (input_x.ndim==1):
        input_x = np.insert(input_x, 0, 1)


    return np.array(input_x)

input1 = np.array([[2,3,6,9],[4,5,7,10]])
input2 = np.array([2,3,6])
input3 = np.array([[2,4],[3,5],[6,7],[9,10]])

for i in [input1, input2, input3]:
    print(x_preprocess(i), "\n")
"""
# -->        [[ 1.  2.  4.]
              [ 1.  3.  5.]
              [ 1.  6.  7.]
              [ 1.  9. 10.]] 

            [1 2 3 6] 

            [[ 1.  2.  4.]
             [ 1.  3.  5.]
             [ 1.  6.  7.]
             [ 1.  9. 10.]] 
"""
def calculate_map_coefficients(aug_x, output_y, lambda_param, sigma):

    X = aug_x
    Y = output_y
    output_df = pd.DataFrame()

    Xt = X.transpose()
    lI = lambda_param*np.identity(len(Xt))
    sS = sigma ** 2
    lpSS = lI*sS
    mmInv = np.linalg.inv(np.dot(Xt, X)+lpSS)
    output_df = np.dot(np.dot(mmInv, Xt), Y)
    weights = np.array(output_df)
    coefs = weights
    
    return coefs

output_y = np.array([208500, 181500, 223500, 
                             140000, 250000, 143000, 
                             307000, 200000, 129900, 
                             118000])
                             
aug_x = np. array([[   1., 1710., 2003.],
                    [   1., 1262., 1976.],
                    [   1., 1786., 2001.],
                    [   1., 1717., 1915.],
                    [   1., 2198., 2000.],
                    [   1., 1362., 1993.],
                    [   1., 1694., 2004.],
                    [   1., 2090., 1973.],
                    [   1., 1774., 1931.],
                    [   1., 1077., 1939.]])
                    
lambda_param = 0.01

sigma = 1000

map_coef = calculate_map_coefficients(aug_x, output_y, 
                                        lambda_param, sigma)
                                        
ml_coef = calculate_map_coefficients(aug_x, output_y, 0,0)

print(map_coef)
# --> np.array([-576.67947107   77.45913349   31.50189177])

print(ml_coef)
#--> np.array([-2.29223802e+06  5.92536529e+01  1.20780450e+03])