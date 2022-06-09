# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 17:47:04 2022

@author: Suklesh
"""
import pandas as pd
data=pd.read_csv('F:\\ExcelR\\Assignments\\Basic Stats Level 1\\Cars.csv')
data.shape
list(data)
data['MPG'].describe()
data['MPG'].hist()
data['MPG'].skew()

#As the data follows normal distribution, we are importing scipy package
from scipy import stats
nd=stats.norm(34.422076,9.131445) #Syntax is .norm(mean,stddev)

#a)P(MPG>38)
k1=nd.cdf(38) #Pyhton will calculate area from left side under the std normal dist curve 
k2=1-k1
print(k2)
#Inference: 34.75% of cars have MPG>38

#b)P(MPG<40)
k3=nd.cdf(40)
print(k3)
#Inference: 72.93% of cars have MPG>38

#c)P(20<MPG<50)
k4=nd.cdf(50)
k5=nd.cdf(20)
k6=k4-k5
print(k6)
#Inference: 89.88% of cars have MPG between 20 and 50

import pandas as pd
data=pd.read_csv('F:\\ExcelR\\Assignments\\Basic Stats Level 1\\wc-at.csv')
data.shape
list(data)
data['AT'].describe()
data['AT'].hist()
data['AT'].skew()

data['Waist'].describe()
data['Waist'].hist()
data['Waist'].skew()







'''
(data['MPG']>38).value_counts()

(data['MPG']<40).value_counts()

(data['MPG']>20).value_counts()

(data['MPG']<50).value_counts()

data_f1=data[(data['MPG']>20) & (data['MPG']<50)]

data_f1["MPG"].describe()
'''







