# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 23:40:41 2022

@author: Suklesh
"""
#KNN for glass data
import pandas as pd
data=pd.read_csv('F:\\ExcelR\\Assignments\\KNN\\glass.csv')
data.shape
data.head
list(data)
data.dtypes
X=data.iloc[:,0:9]
X.hist()
X.shape

Y=data['Type']

#Standardising the data
from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
X_scale = SS.fit_transform(X)
print(X_scale)
X_scale=pd.DataFrame(X_scale)

#Splitting data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X_scale,Y,test_size=0.25,random_state=95)
X_train.shape
Y_train.shape

#KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7,p=2)
knn.fit(X_train,Y_train)
Y_pred=knn.predict(X_test)

#Metrics
from sklearn.metrics import accuracy_score
acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of KNN with k=7 is:",(acc*100).round(3))

