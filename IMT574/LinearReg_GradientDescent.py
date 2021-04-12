#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 17:19:05 2021

@author: stlp
"""

import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn import preprocessing

#flight =  pd.read_csv('data/airline_costs.csv')    

flight_cols = ['Airline', 'length','speed', 'flighttime', 'population', 'operating_cost'
               , 'revenue','ton_mile','capacity','asset','investment','adj_assets']


flight = pd.read_csv('data/airline_costs.csv', header=None, names = flight_cols)
flight

X = flight[['length','flighttime']] # Adds constant/intercept term
X = sm.add_constant(X) # Adds constant/intercept term
y = flight[['population']]

lr_model = sm.OLS(y,X).fit()

print(lr_model.summary())

# Line equation: population = -7792.0706 + (183.2956*length) +(-213.3340 * flighttime)

# 36659.12 - 1536.0048 - 7792.0706

# 27331.0446 * 1000 = 27331044

X2 = flight[['population']] # Adds constant/intercept term
X2 = sm.add_constant(X2) # Adds constant/intercept term
y2 = flight[['asset']] 

lr_model2 = sm.OLS(y2,X2).fit()

print(lr_model2.summary())

# Line equation: asset = -98.5080  + (0.0217 * popu)

# 36659.12 -98.5080

# 342.002 * 100000

## Question 2 

kang = pd.read_excel('data/slr07.xls')
kang

X3 = kang[['X']]
X3 = sm.add_constant(X3)
y3 = kang[['Y']]



lr_model3 = sm.OLS(y3,X3).fit()
print(lr_model3.summary())


def grad_descent(X, y, alpha, epsilon):    
    iteration = [0]    
    i = 0    
    theta = np.ones(shape=(len(kang.columns), 1))    
    cost = [np.transpose(X @ theta - y) @ (X @ theta - y)]    
    delta = 1    
    while (delta>epsilon):        
        theta = theta - alpha*((np.transpose(X)) @ (X @ theta - y))        
        cost_val = (np.transpose(X @ theta - y)) @ (X @ theta - y)        
        cost.append(cost_val)        
        delta = abs(cost[i+1]-cost[i])        
        if ((cost[i+1]-cost[i]) > 0):            
            print("The cost is increasing. Try reducing alpha.")            
            break        
        iteration.append(i)        
        i += 1    
        
    print("Completed in %d iterations." %(i))    
    return(theta)

X3 = pd.concat((pd.DataFrame([1,len(kang)]*24), 
               kang[['X']]),axis=1, join='outer').to_numpy()

y3 = y3.to_numpy()
theta = grad_descent(X = preprocessing.scale(X3), y=y3, alpha=0.02, epsilon = 10**-10)

print (theta)


