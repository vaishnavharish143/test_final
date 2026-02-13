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

# print(x_train.shape)

x_train_age = x_train.drop(columns =
                            ['sex','bmi','children','smoker','region','charges']).values

x_test_age = x_test.drop(columns =
                         ['sex','bmi','children','smoker','region','charges']).values

# print(x_train_age.shape)

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn. compose import ColumnTransformer

transformer = ColumnTransformer(transformers = [
    ('tnf1', OrdinalEncoder(categories =[['no','yes']]), ['smoker']),
    ('tnf2', OneHotEncoder(sparse_output = False, drop = 'first'), ['sex','region'])],
    remainder='passthrough')
print(transformer.fit_transform(x_train).shape)
print(transformer.fit_transform(x_test).shape)


