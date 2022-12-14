import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle
import os

df = pd.read_csv('breast-cancer.csv')

encoder = LabelEncoder()
df.diagnosis=encoder.fit_transform(df.diagnosis)
scaler = MinMaxScaler()
df.drop('id',axis=1,inplace=True)
y = df['diagnosis']
x = df.drop('diagnosis',axis=1)
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=5)
model = LogisticRegression()
model.fit(xtrain,ytrain)

pickle.dump(model, open('Logestic.pkl', 'wb'))




