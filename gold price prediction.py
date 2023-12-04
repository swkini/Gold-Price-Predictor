# %% [markdown]
# ## GOLD PRICE PREDICTION

# %% [markdown]
# #### Importing the libraries

# %%
import pandas as pd 
import numpy as nm
import matplotlib.pyplot as mtp
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

# %% [markdown]
# #### Data Collection And Processing

# %%
# loading a csv data to a pandas dataframe
df=pd.read_csv('gld_price_data.csv')

# %%
# print first 4 rows of the data
df .head()

# %%
# print last 4 rows of the data
df .tail()

# %% [markdown]
# SPX- free-float weighted measurement stock market index of 500 largest comapanies listed on stock exchanges in the US
# 

# %% [markdown]
# GLD- Gold Price

# %% [markdown]
# USO- United States Oil Fund

# %% [markdown]
# SLV- Silver Price

# %% [markdown]
# EUR/USD- currency pair quotation of the Euro against the US

# %%
# no of rows and columns
df.shape

# %%
# checking the number of missing values
df.isnull().sum()

# %%
# getting the statistical information of the data 
df.describe()

# %%
# Create a list of columns to exclude (e.g., date columns)
columns_to_exclude = ['Date']  # Replace 'Date' with the actual column name to exclude

# Exclude the specified columns
numeric_df = df.drop(columns=columns_to_exclude)

# Calculate correlations for the remaining numeric columns
correlation_matrix = numeric_df.corr()


# %%
# constructing a heatmap to understand correlation
mtp.figure(figsize=(8,8))
sns.heatmap(correlation_matrix, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8},cmap='Blues')

# %%
print(correlation_matrix['GLD'])

# %%
# Checking the distrubution og the GLD price
sns.distplot(df['GLD'])

# %% [markdown]
# Splitting the Features and Target

# %%
#feature set
X = df.drop(['Date','GLD'],axis=1)

# %%
#target set
Y=df['GLD']

# %%
print(Y)

# %%
print(X)

# %%
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)


# %% [markdown]
# Model Training:Random Forest Regressor

# %%
regressor1=RandomForestRegressor(n_estimators=100)

# %%
#training the model
regressor1.fit(X_train,Y_train)

# %%
#prediction on test data
pred1=regressor1.predict(X_test)

# %%
print(pred1)

# %%
#R square error
error_score1=metrics.r2_score(Y_test,pred1)
print("R error_score:",error_score1)

# %% [markdown]
# Compare the Actual Values and Predicted Value in a Plot

# %%
Y_test=list(Y_test)

# %%
mtp.plot(Y_test,color="black",label="Actual Value")
mtp.plot(pred1,color="green",label="Predicted Value")
mtp.title("Actual Price vs Predicted Price")
mtp.xlabel("Number of vales")
mtp.ylabel("GLD Price")
mtp.legend()
mtp.show()

# %%
#train and test score for random forest regressor
print("Train Score: ",regressor1.score(X_train,Y_train))
print("Test Score: ",regressor1.score(X_test,Y_test))

# %% [markdown]
# Model Training: Linear Regression

# %%
regressor2=LinearRegression()

# %%
#training the model
regressor2.fit(X_train,Y_train)

# %%
#prediction on test data
pred2=regressor2.predict(X_test)

# %%
print(pred2)

# %%
#R square error
error_score2=metrics.r2_score(Y_test,pred2)
print("R error_score:",error_score2)

# %% [markdown]
# Compare the Actual Values and Predicted Value in a Plot

# %%
mtp.plot(Y_test,color="black",label="Actual Value")
mtp.plot(pred2,color="pink",label="Predicted Value")
mtp.title("Actual Price vs Predicted Price")
mtp.xlabel("Number of vales")
mtp.ylabel("GLD Price")
mtp.legend()
mtp.show()

# %%
#train and test score for linear regression
print("Train Score: ",regressor2.score(X_train,Y_train))
print("Test Score: ",regressor2.score(X_test,Y_test))

# %% [markdown]
# Model Training: Decision Tree Regressor

# %%
regressor3=DecisionTreeRegressor()

# %%
#training the model
regressor3.fit(X_train,Y_train)

# %%
#prediction on test data
pred3=regressor3.predict(X_test)

# %%
print(pred3)

# %%
#R square error
error_score3=metrics.r2_score(Y_test,pred3)
print("R error_score:",error_score3)

# %% [markdown]
# Compare the Actual Values and Predicted Value in a Plot

# %%
mtp.plot(Y_test,color="black",label="Actual Value")
mtp.plot(pred3,color="red",label="Predicted Value")
mtp.title("Actual Price vs Predicted Price")
mtp.xlabel("Number of vales")
mtp.ylabel("GLD Price")
mtp.legend()
mtp.show()

# %%
#train and test score for decision tree regressor
print("Train Score: ",regressor3.score(X_train,Y_train))
print("Test Score: ",regressor3.score(X_test,Y_test))

# %% [markdown]
# ## Predicting with user input

# %%
def predictor(spx,uso,slv,quo):
    x1=nm.array([[spx,uso,slv,quo]])
    feature_names=X_train.columns
    input_dict={name:value for name,value in zip(feature_names,x1.flatten())}
    input_df=pd.DataFrame([input_dict])
    p=regressor3.predict(input_df)
    print("The gold price using decision tree regressor is: ",p)

# %%
spx=float(input("Enter the SPX (example:1400): "))
uso=float(input("Enter the USO (example:70): "))
slv=float(input("Enter the Silver price (example:20): "))
quo=float(input("Enter the EUR/USD quotation (example:1.1/1.2...): "))
predictor(spx,uso,slv,quo)


# %%



