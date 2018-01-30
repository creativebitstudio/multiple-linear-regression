
# coding: utf-8

# # Multiple Linear Regression

# #### Multiple linear regression: 
# 1. Is a <b>predicitve</b> linear regression analysis model.
# 2. Multiple in multiple linear regression (MLR) refers to two or more predictors, unlike simple linear regression that refers to one and only one predictor. 
# 2. We use MLR to explain the relationship between one continues dependent variable (also known as the response variable) and two or more independent variables (also known as explanatory variables) in a <b>linear</b> approach model.
# 3. The independent variables can be continuous or categorical (dummy coded as appropriate)
# 4. When we encounter a large number of independent variables, it is more efficient to use matrices (or matrix, singular) to define the regression model and the subsequent analyses.
# 5. A matrix (or matrices, plural) is a rectangular or square array of elements (usually numbers) arranged in rows and columns. It is called a dimension of matrix when it is expressed as the number of rows multiplied by the number of columns. A vector, however, is a matrix with only one row, or one column. Arithmetic operations can be performed amongst matrices all the way to using the Laplace formula and more. Advanced topics include range, nullspace, and projections. Gauss-Jordan elimination. And Eigendecomposition         
#     
#     a) Example of a matrix
#     
# <img src="Matrix.svg.png", width=220, height=220>

# ### Problem: Find the startups with the most return in investment. 
# 
# y = profit - our dependent varibale.
# x1, x2, x3 and x4 are independent varibales.
# 
# Note: we will be using dummy varibales for the State Column, since we have three different and repetitive categories. New York, Florida and California. Therefore, both categories will be represented as a boolean, 1's and 0's.
# 
# Multiple Linear Regression Formula with Dummy Varibale: y = b0 + b1 * x1 + ... + b4 * x4
# 
# Note: in the case were we have 2 or more dummy varibale columns, ommit all except the first one. Use that with the formula and in code.

# In[1]:


# Import Python Libraries
import numpy as np
import pandas as pd
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt


# ### Import the Dataset

# In[2]:


dataset = pd.read_csv('50_Startups.csv')
dataset # shows the whole datset in a nice table format


# ### Obtain the independent variables of X and separate columns from the dataset to obtain the independent variable of X. In this case we take all columns but the last one (Profit).

# In[3]:


# iloc works on the positions in the index (takes in integers as parameters).
X = dataset.iloc[:,:-1].values # x is matrix of features, the independent variable.
print(X) # prints the independent variables .


# ### Now lets obtain the dependent variables of y

# In[4]:


# Y is set to the Profit values (dependent variable).
y = dataset.iloc[:,4].values
print(y) # prints the y values for profit.


# ### Encode categorical data for the independent variable of X

# In[5]:


# Encoding categorical data with OneHotEncoder.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3]) 
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()


# Once encoded, the values of X will include 3 dummy varibales within their own columns. Index 0 (column 1) is California, index 1 (column 2) is Florida, and index 2 is New York (column 3).

# In[6]:


print(X) # prints X variables   


# The following code lets us remove the unnecessary values, once the dummy variables that were produced earlier from our state column in the dataset. We will also see how the formula y = b0 + b1*X1... is represented by each column from left to right values.

# In[7]:


# Removing the dummy varibale not needed   
X = X[:, 1:] # take every column starting from index 1 onwards
print(X) # prints x values


# ### Spliting the dataset into the Training set and Test set

# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[9]:


# Uncomment one of the following to view the values of each train and test set for X and y
# These are the real values before our model does the predicition
# X_train
# X_test
# y_train
# y_test


# ### Fitting Multiple Linear Regression to the Training Set

# In[10]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# ### Predicting the Test set results

# In[11]:


y_pred = regressor.predict(X_test)
print(y_pred)


# ### Building the optimal model using backward elimination 

# In[12]:


# import library
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# X_opt = X[:, [0, 1, 3, 4, 5]]
# regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# regressor_OLS.summary()
# X_opt = X[:, [0, 3, 4, 5]]
# regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# regressor_OLS.summary()
# X_opt = X[:, [0, 3, 5]]
# regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# regressor_OLS.summary()


# Note: discard the backward elimination model if it loks a bit confusing, look up backward elimination for a multiple linear regression model to better understand it.
