# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 17:49:26 2022

@author: Suklesh
"""

import pandas as pd
data=pd.read_csv("F:\\ExcelR\\Assignments\\Basic Stats Level 1\\Q9_a.csv")
list(data)
data.shape
#a)Speed
data['speed'].kurtosis()
data['speed'].skew()

#b)Distance
data['dist'].kurtosis()
data['dist'].skew()

data_1=pd.read_csv('F:\\ExcelR\\Assignments\\Basic Stats Level 1\\Q9_b.csv')
list(data_1)
#a)SP
data_1['SP'].kurtosis()
data_1['SP'].skew()

#a)WT
data_1['WT'].kurtosis()
data_1['WT'].skew()
