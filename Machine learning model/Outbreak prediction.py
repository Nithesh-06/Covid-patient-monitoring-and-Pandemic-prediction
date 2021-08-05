# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 12:31:34 2020

@author: Nithesh C
"""
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv('corona.csv') #reading the dataset
X=dataset.iloc[:, 0:1].values #creating a matrix X
y=dataset.iloc[:, 1].values #creating a vector Y

#splitting the dataset into training set and test set
#from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=1/3,random_state=0)

# Fitting Linear Regression to the Dataset
from sklearn.linear_model import LinearRegression
'''lin_reg = LinearRegression()
lin_reg.fit(X, y)'''

# Fitting the Polynomial Regression to the Dataset
from sklearn.preprocessing import PolynomialFeatures #importing library
poly_reg=PolynomialFeatures(degree=6) #Object used to create a matrix with the powers of X
X_poly=poly_reg.fit_transform(X) #Creating a matrix with the powers of X

lin_reg_2=LinearRegression() #Object for Linear regression
lin_reg_2.fit(X_poly,y) #fitting the linear regression model

# Visualising the Linear Regression Model
'''plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or Bluff (Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()'''

#Visualising the Polynomial Regresssion Model
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue') # Here the second arguement's argument must be a matrix that contains powers of X 
plt.title('Polynomial Regression')
plt.xlabel('Day')
plt.ylabel('No of cases')
plt.show() 

#Predicting a new result with linear regression
#lin_reg.predict([[6.5]]) 
#Predicting a new result with Polynoiam Regression
lin_reg_2.predict(poly_reg.fit_transform([[83]]))
