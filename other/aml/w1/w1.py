import os
data_path =  os.path.abspath(os.path.join('other','aml','w1','datasets'))

### This cell imports the necessary modules and sets a few plotting parameters for display

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)

### Read in the data
tr_path = data_path+ '/train.csv'
test_path = data_path+'/test.csv'
data = pd.read_csv(tr_path)

### The .head() function shows the first few lines of data for perspecitve
data.head()

### List the column names
data.columns

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


### price v. year
### Using Pandas

data.plot('YearBuilt', 'SalePrice', kind = 'scatter', marker = 'x');

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



    
### GRADED
### Build a function called "select_columns"
### As inputs, take a DataFrame and a *list* of column names.
### Return a DataFrame that only has the columns specified in the list of column names
### Grading will check type of object, dimensions of object, and column names
### YOUR ANSWER BELOW

def select_columns(data_frame, column_names):
    return data_frame[column_names]


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
    inv_m1 = m1.transpose()
    mm = m1.T.dot(m1)
    mm = np.linalg.inv(np.asmatrix(mm,dtype=int))
    mv = m1.T.dot(m2)
    output_df = mm.dot(mv)
    
    #output_df = inverse_of_matrix(inv_m1.dot(m1)).dot(inv_m1).dot(m2)

    return output_df


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