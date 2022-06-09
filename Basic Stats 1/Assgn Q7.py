# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 13:02:45 2022

@author: Suklesh
"""

import pandas as pd
Q7=pd.read_csv("F:\\ExcelR\\Assignments\\Basic Stats Level 1\\Q7.csv")
Q7.shape
list(Q7)

#Mean, Median, Mode, Variance, Standard deviation, range of 

#a) Points
import statistics
Q7['Points'].mean()
Q7['Points'].median()
Q7['Points'].mode()
Q7['Points'].min()
Q7['Points'].max()
Q7['Points'].var()

#Or
Q7['Points'].describe()

#b) Score
import statistics
Q7['Score'].mean()
Q7['Score'].median()
Q7['Score'].mode()
Q7['Score'].min()
Q7['Score'].max()
Q7['Score'].var()

#Or
Q7['Score'].describe()

#c) Weigh
import statistics
Q7['Weigh'].mean()
Q7['Weigh'].median()
Q7['Weigh'].mode()
Q7['Weigh'].min()
Q7['Weigh'].max()
Q7['Weigh'].var()

#Or
Q7['Weigh'].describe()




