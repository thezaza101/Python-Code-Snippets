
# coding: utf-8

# # Linear Regression - Least Squares  
# 
# -------------  
# 
# _Author: Khal Makhoul, W.P.G.Peterson_  
# 
# ## Project Guide
# ----------------  
# - [Project Overview](#overview)
# - [Introduction and Review](#intro)
# - [Data Exploration](#data)  
# - [Coding Linear Regression](#code)
# 
# <a id = "overview"></a>
# ## Project Overview
# ----------
# #### EXPECTED TIME 2 HRS
# 
# This assignment will test your ability to code your own version of least squares regression in `Python`. After a brief review of some of the content from the lecture you will be asked to create a number of functions that will eventually be able to read in raw data to `Pandas` and perform a least squares regression on a subset of that data.  
# 
# This will include:  
# - Calculating least squares weights
# - reading data on dist to return `Pandas` DataFrame  
# - select data by column  
# - implement column cutoffs  
# 
# ** Motivation**: Least squares regression offer a way to build a closed-form and interpretable model.  
# 
# **Objectives**: This assignment will:
# - Test `Python` and `Pandas` competency
# - Ensure understanding of the mathematical foundations behind least squares regression  
# 
# **Problem**: Using housing data, we will attempt to predict house price using living area with a regression model.  
# 
# **Data**: Our data today comes from [Kaggle's House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).  
# 
# See above link for Description of data from `Kaggle`.  
# 
# <a id = "intro"></a>
# ### Introduction and Review
# 
# As long as a few basic assumptions are fulfilled, linear regression using least squares is solvable exactly, without requiring approximation. 
# 
# This means that the equations presented in the week 1 lectures can be adapted directly to `Python` code, making this good practice both for using `Python` and translating an "algorithm" to code.
# 
# We will use the matrix version of the least squares solution presented in lecture to derive the desired result. As a reminder, this expresses the least squares coefficients $w_{LS}$ as a vector, and calculates that vector as a function of $X$, the matrix of inputs, and $y$, the vector of outputs from the training set:
# 
# $$w_{LS} = (X^T X)^{−1}X^T y,$$
# 
# where $w_{LS}$ refers to the vector of weights we are trying to find, $X$ is the matrix of inputs, and $y$ is the output vector. 
# 
# In this equation, $X$ is always defined to have a vector of $1$ values as its first column. In other words, even when there is only one input value for each data point, $X$ takes the form:
# 
# $$
# X = \begin{bmatrix}
# 1 \  x_{11}  \\
# 1 \  x_{21}  \\
# \vdots \ \vdots \\
# 1 \ x_{n1}
# \end{bmatrix} 
# $$
# 
# For two inputs per data point, $X$ will take this form:
#  
# $$
# X = \begin{bmatrix}
# 1 \  x_{11} \  x_{12} \\
# 1 \  x_{21} \  x_{22} \\
# \vdots \ \vdots \\
# 1 \ x_{n1} \  x_{n2}
# \end{bmatrix} 
# $$
# 
# 
# 
# 
# Please refer to lecture notes for additional context.  
# <a id = "data"></a>
# ### Data Exploration
# 
# Before coding an algorithm, we will take a look at our data using `Python`'s `pandas`. For visualizations we'll use `matplotlib`. Familiarity with these modules will serve you well. The following cells include comments to explain the purpose of each step.

# In[1]:


### This cell imports the necessary modules and sets a few plotting parameters for display

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)


# In[2]:


### Read in the data
tr_path = '../resource/asnlib/publicdata/train.csv'
test_path = '../resource/asnlib/publicdata/test.csv'
data = pd.read_csv(tr_path)


# In[3]:


### The .head() function shows the first few lines of data for perspecitve
data.head()


# In[4]:


### Lists column names
data.columns


# In[5]:


### GRADED
### How many columns are in `data`?
### assign int answer to ans1
### YOUR ANSWER BELOW

ans1 = 81


# In[6]:


#
# AUTOGRADER TEST - DO NOT REMOVE
#


# #### Visualizations

# In[7]:


### We can plot the data as follows
### Price v. living area
### with matplotlib

Y = data['SalePrice']
X = data['GrLivArea']

plt.scatter(X, Y, marker = "x")

### Annotations
plt.title("Sales Price vs. Living Area (excl. basement)")
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice");


# In[8]:


### price v. year
### Using Pandas

data.plot('YearBuilt', 'SalePrice', kind = 'scatter', marker = 'x');


# In[9]:


### GRADED
### Given the above graphs, it appears there is a:  
### True) positive correlation between the variables
### False) negative correlation between the variables
### Assign boolean corresponding to choice to ans1
### YOUR ANSWER BELOW

ans1 = 'True'


# In[10]:


#
# AUTOGRADER TEST - DO NOT REMOVE
#


# ### Submission Instructions
# 
# You will have to ensure that the function names match the examples provided.
# 
# The code will be automatically graded by a script that will take your code as input and execute it. In order for the grading script to work properly, you must follow the naming conventions in this assignment stub.  
# 
# <a id = "code"></a>
# ### Coding Linear Regression
# Given the equation above for $w_{LS}$, we know all we need to in order to solve a linear regression. Coding out the steps in `Python`, we will complete the process in several steps.
# 
# #### Matrix Operations
# Below is an example of a function that takes the inverse of a matrix. The `numpy` module is used, and all the function does is call the `numpy` function `np.linalg.inv()`. Though simple, this can be used as a template for a few good coding practices:
# 
# * Name functions and parameters descriptively
# * Use underscores _ to separate words in variable/function names (snake_case, **NOT** PascalCase or camelCase)
# * In functions and classes, include a docstring between triple quotes 

# In[11]:


### GRADED
### Build a function that takes as input a matrix
### return the inverse of that matrix
### assign function to "inverse_of_matrix"
### YOUR ANSWER BELOW

def inverse_of_matrix(mat):
    matrix_inverse = np.linalg.inv(mat)
    return matrix_inverse

### Testing function:

print("test",inverse_of_matrix([[1,2],[3,4]]), "\n")
print("From Data:\n", inverse_of_matrix(data.iloc[:2,:2]))


# In[12]:


#
# AUTOGRADER TEST - DO NOT REMOVE
#


# #### Q1: Read Data

# In[13]:


### GRADED
### In order to create any model it is necessary to read in data
### Build a function called "read_to_df" that takes the file_path of a .csv file.
### Use a pandas functions appropriate for .csv files to turn that path into a DataFrame
### Use pandas function defaults for reading in file
### Return that DataFrame
### Grade will be determined by whether or not the returned item is of type "DataFrame" and
### if the dimensions are correct
### YOUR ANSWER BELOW
import pandas as pd
def read_to_df(file_path):    
    return pd.read_csv(file_path)


# In[14]:


#
# AUTOGRADER TEST - DO NOT REMOVE
#


# #### Q2: Select by Columns

# In[15]:


### GRADED
### Build a function called "select_columns"
### As inputs, take a DataFrame and a *list* of column names.
### Return a DataFrame that only has the columns specified in the list of column names
### Grading will check type of object, dimensions of object, and column names
### YOUR ANSWER BELOW

def select_columns(data_frame, column_names):
    return data_frame[column_names]


# In[16]:


#
# AUTOGRADER TEST - DO NOT REMOVE
#


# #### Q2a:

# In[17]:


### GRADED
### For a `Pandas` DataFrame named `df`, the names of columns may be accessed by the:
### `df.columns` attribute.
### The names of the rows may be accessed by the `df.<ans1>` attribute
### to ans1 assign a string that when placed after `df.` will return the row names
### of a DataFrame
### YOUR ANSWER BELOW

ans1 = 'index'


# In[18]:


#
# AUTOGRADER TEST - DO NOT REMOVE
#


# #### Q3: Subset Data by Value

# In[35]:


### GRADED
### Build a function called "column_cutoff"
### As inputs, accept a Pandas Dataframe and a list of tuples.
### Tuples in format (column_name, min_value, max_value)
### Return a DataFrame which excludes rows where the value in specified column exceeds "max_value"
### or is less than "min_value"
### ### NB: DO NOT remove rows if the column value is equal to the min/max value
### YOUR ANSWER BELOW

def column_cutoff(data_frame, cutoffs):    
    df_cutoff = data_frame
    for lim in cutoffs: 
        
        #df3 = result[result['Value'] > 10]  
        df_cutoff = df_cutoff[(df_cutoff[str(lim[0])] >= lim[1]) & (df_cutoff[str(lim[0])] <= lim[2])]
        
        #df_cutoff = df_cutoff.loc[(df_cutoff[lim[0]] >=lim[1]) & df_cutoff[lim[0]]<=lim[2]]
    return df_cutoff


# In[27]:


#
# AUTOGRADER TEST - DO NOT REMOVE
#


# Next you'll implement the equation above for $w_{LS}$ using the inverse matrix function.  
# $$w_{LS} = (X^T X)^{−1}X^T y,$$
# 
# #### Q4: Least Squares

# In[21]:


### GRADED
### Build a function  called "least_squares_weights"
### take as input two matricies corresponding to the X inputs and y target
### assume the matricies are of the correct dimensions

### Step 3: Use the above equation to calculate the least squares weights.

### NB: `.shape`, `np.matmul`, `np.linalg.inv`, `np.ones` and `np.transpose` will be valuable.
### If those above functions are used, the weights should be accessable as below:  
### weights = least_squares_weights(train_x, train_y)
### weight1 = weights[0][0]; weight2 = weights[1][0];... weight<n+1> = weights[n][0]

### YOUR ANSWER BELOW

def least_squares_weights(input_x, target_y):
    m1 = input_x
    m2 = target_y    
    
    # Step 1
    if (len(m1) < m1.shape[1]):
        m1 = input_x.transpose()
    if (len(m2) < m2.shape[1]):
        m2 = target_y.transpose()
    
    # Step 2
    m1 = np.concatenate((np.ones((m1.shape[0],1), dtype=int), m1), axis=1)
        
    # Step 3
    #np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y)) # one method
    inv_m1 = m1.transpose()
    #mm = inv_m1.dot(m1)
    #mv = inv_m1.dot(m2)
    mm = m1.T.dot(m1)
    mm = np.linalg.inv(np.asmatrix(mm,dtype=int))
    mv = m1.T.dot(m2)
    output_df = mm.dot(mv)
    
    #output_df = inverse_of_matrix(inv_m1.dot(m1)).dot(inv_m1).dot(m2)

    return output_df


# In[22]:


#
# AUTOGRADER TEST - DO NOT REMOVE
#


# In[23]:


### GRADED
### Why, in the function  above, is it necessary to prepend a column of ones
### 'a') To re-shape the matrix
### 'b') To create an intercept term
### 'c') It isn't needed, it's just meant to be confusing
### 'd') As a way to make sure the weights turn out positive
### Assign the character asociated with your choice as a string to ans1
### YOUR ANSWER BELOW

ans1 = 'a'


# In[24]:


#
# AUTOGRADER TEST - DO NOT REMOVE
#


# #### Testing on Real Data
# 
# Now that we have code to read the data and perform matrix operations, we can put it all together to perform linear regression on a data set of our choosing.  
# 
# If your functions above are defined correctly, the following two cells should run without error.

# In[30]:


df = read_to_df(tr_path)
df_sub = select_columns(df, ['SalePrice', 'GrLivArea', 'YearBuilt'])

cutoffs = [('SalePrice', 50000, 1e10), ('GrLivArea', 0, 4000)]
df_sub_cutoff = column_cutoff(df_sub, cutoffs)

X = df_sub_cutoff['GrLivArea'].values
Y = df_sub_cutoff['SalePrice'].values

### reshaping for input into function
training_y = np.array([Y])
training_x = np.array([X])

weights = least_squares_weights(training_x, training_y)
print(weights)


# In[31]:


max_X = np.max(X) + 500
min_X = np.min(X) - 500

### Choose points evenly spaced between min_x in max_x
reg_x = np.linspace(min_X, max_X, 1000)

### Use the equation for our line to calculate y values
reg_y = weights[0][0] + weights[1][0] * reg_x

plt.plot(reg_x, reg_y, color='#58b970', label='Regression Line')
plt.scatter(X, Y, c='k', label='Data')

plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.legend()
plt.show()


# #### Model Evalutaion Intro
# Further lessons will discuss model evaluation scores in more detail, quickly, here we will calculate root mean squared errors with our calculated weights

# In[26]:


### GRADED
### True or False
### The Root Mean Square Error is in the same units as the data
### assign boolean response to ans1
### YOUR ANSWER BELOW

ans1 = 'True'


# In[ ]:


#
# AUTOGRADER TEST - DO NOT REMOVE
#


# #### Calculating RMSE

# In[2]:


rmse = 0

b0 = weights[0][0]
b1 = weights[1][0]

for i in range(len(Y)):
    y_pred = b0 + b1 * X[i]
    rmse += (Y[i] - y_pred) ** 2
rmse = np.sqrt(rmse/len(Y))
print(rmse)


# #### Calculating $R^2$

# In[ ]:


ss_t = 0
ss_r = 0

mean_y = np.mean(Y)

for i in range(len(Y)):
    y_pred = b0 + b1 * X[i]
    ss_t += (Y[i] - mean_y) ** 2
    ss_r += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_r/ss_t)

print(r2)


# ### sklearn implementation
# 
# While it is useful to build and program our model from scratch, this course will also introduce how to use conventional methods to fit each model. In particular, we will be using the `scikit-learn` module (also called `sklearn`.)  
# 
# Check to see how close your answers are!

# In[1]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()

### sklearn requires a 2-dimensional X and 1 dimensional y. The below yeilds shapes of:
### skl_X = (n,1); skl_Y = (n,)
skl_X = df_sub_cutoff[['GrLivArea']]
skl_Y = df_sub_cutoff['SalePrice']

lr.fit(skl_X,skl_Y)
print("Intercept:", lr.intercept_)
print("Coefficient:", lr.coef_)


# In[ ]:




