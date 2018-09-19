
# coding: utf-8

# In[67]:


# The aim of this project is to predict the number of bikes rented out by people
# in a given hour. The data used isfrom UC Irvine's website which contains
# a number of features related to bike rentals in Washington DC. 
# This project is to practise the algorithm Decision Trees and Random
# Forest Classifiers. 


# In[68]:


# Importing a few required libraries, importing the data and running some commands
# to get a jist of the dataset. 
import pandas as pd
import numpy as np 

bike_rentals = pd.read_csv("bike_rental_hour.csv")
print(bike_rentals.head(5))


# In[69]:


print(bike_rentals.info())


# In[70]:


# The data looks consistent. Now to visualize what I have. 
import matplotlib.pyplot as plt 
get_ipython().magic('matplotlib inline')

plt.hist(bike_rentals["cnt"])
plt.show()


# In[71]:


# This shows the right skewed distributuon of the counts. 
# I'll generate the correlations to see which columns are most closely related 
# to the count
correlations = bike_rentals.corr()["cnt"]
correlations = correlations.sort_values(ascending = False)
print(correlations)


# In[72]:


# Reading the documentation, it seems like the 'hr' column cosists of a 24 hour 
# format. It would be interesting to see if the time of day is somehow related to 
# the rentals. I will split the 'hr' into 4 different categories to make that
# observation. 

# Defining the function that will enable me to do so. 
def assign_label(hour):
    if hour>=0 and hour<6:
        return 1
    elif hour>=6 and hour<12:
        return 2
    elif hour>=12 and hour<18:
        return 3
    elif hour>=18 and hour<=24:
        return 4 
bike_rentals["time_label"] = bike_rentals["hr"].apply(assign_label)
print(bike_rentals["time_label"].head(10))


# In[73]:


# Now to divide the dataframe into train/test dataframes for training and testing

# I will use a 80:20 ratio split. 
split_ratio = np.floor(0.8*len(bike_rentals))
split_ratio = int(split_ratio)
train = bike_rentals.sample(n = split_ratio)
bike_rentals_copy = bike_rentals.copy()
test = bike_rentals_copy.drop(train.index)

# Checking the length of each dataframe to see if it the split is correct 
print(len(bike_rentals))
print(len(train))
print(len(test))


# In[74]:


# Some columns in the dataset are a subset of the 'count' and hence need ot be dropped.
# the 'dteday' is the date of the purchase and is laos irrelevent. 
train = train.drop(columns = ["casual","registered", "dteday"])
print(train.columns)


# In[76]:


# Creating the fetaure list for regression. 
features = train.columns
features = features.drop("cnt")
print(features)


# In[79]:


# Importing the encessary libraries. 
# I will use the mean_squared_error metric to asses the error since its a regression
# and this will perhaps be the most relevant 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Tthe algorithm for train/test
lr = LinearRegression()
lr.fit(train[features],train["cnt"])
predictions = lr.predict(test[features])

mse = mean_squared_error(test["cnt"],predictions)
print(mse)


# In[91]:


# Now to compare this error with that of a DecisionTree 
from sklearn.tree import DecisionTreeRegressor
values=dict()
for i in range(1,20,1):
    dtr = DecisionTreeRegressor(min_samples_leaf = i)
    dtr.fit(train[features],train["cnt"])
    predictions = dtr.predict(test[features])
    mse = mean_squared_error(test["cnt"],predictions)
    values[i] = mse
#     print(mse)
best_i = min(values, key =values.get)
print(best_i,values[best_i])


# In[93]:


# Here the error has reduced from 18000+ to around 2500 which is a significant
# change. However, there is scope to reduce it further if I use a RandomForest

# Algorithm for Random Forest
from sklearn.ensemble import RandomForestRegressor 

# I will also try to optimize both 'min samples leaf' and 'n estimators' parameters
# Optimizing min_samples_leaf
vals = dict()
for i in range(1,25):
    rfr = RandomForestRegressor(min_samples_leaf=i)
    rfr.fit(train[features],train["cnt"])
    predictions = rfr.predict(test[features])
    mse = mean_squared_error(test["cnt"],predictions)
    vals[i] = mse
low = min(vals,key=vals.get)
print(low, vals[low])


# In[94]:


number = [100,200,300,400,500]
new_vals = dict()
for j in number:
    rfr = RandomForestRegressor(min_samples_leaf=1,n_estimators = j)
    rfr.fit(train[features],train["cnt"])
    predictions = rfr.predict(test[features])
    mse = mean_squared_error(test["cnt"],predictions)
    new_vals[j] = mse
new_low = min(new_vals,key=new_vals.get)
print(new_low, new_vals[new_low])


# In[ ]:


# As observed, the error has gone down even more to 1497 using the values
# min_samples_leaf = 1 and n_estimators = 500. However this is a slow process. 
# I

