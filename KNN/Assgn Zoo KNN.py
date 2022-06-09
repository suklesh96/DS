# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 20:41:24 2022

@author: Suklesh
"""
#KNN for ZOO dataset
import numpy as np
import pandas as pd
data=pd.read_csv("F:\\ExcelR\\Assignments\\KNN\\Zoo.csv")
data.head()
data.shape
data.dtypes
list(data)

#Splitting data into X and Y 
X=data.iloc[:,1:17]
list(X)
X.head()
X.ndim
Y=data.iloc[:,17:]
Y.head()
X.dtypes

#Standardize the X 
from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
X_scale=SS.fit_transform(X)
print(X_scale)

#Splitting the data into train annd test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X_scale,Y,test_size=0.25,random_state=41)
X_train.shape

#KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5).fit(X_train,Y_train)
Y_pred=knn.predict(X_test)

#Metrics
from sklearn.metrics import accuracy_score
acc=accuracy_score(Y_test,Y_pred)
print(acc)

#Cross validation
tr_err=[]
t_err=[]
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
for i in range(1,101,1):
    X_train,X_test,Y_train,Y_test=train_test_split(X_scale,Y,test_size=0.25,random_state=i)
    knn=KNeighborsClassifier(n_neighbors=5).fit(X_train,Y_train)
    
    Y_pred_tr=knn.predict(X_train)
    Y_pred_t=knn.predict(X_test)
    
    tr_err.append(mean_squared_error(Y_train, Y_pred_tr))
    t_err.append(mean_squared_error(Y_test, Y_pred_t))
    
Tr_err=np.mean(tr_err)
T_err=np.mean(t_err)

print(Tr_err)
print(T_err)










