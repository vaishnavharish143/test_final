import numpy as np
import pandas as pd

df = pd.read_csv("insurance - insurance.csv")
# print(df.head(3))

df = df.dropna()
# print(df.dropna())

x = df.drop(columns= ['charges'])
y = df['charges']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.2, random_state=42)
# print(x_train)

print(x_train.shape)