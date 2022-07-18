import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn import linear_model

df = pd.read_csv('sindh-school-enrollment-stats.csv')
df.rename(columns={'State/UTs': 'STATE', 'Total Cases': 'Cases'}, inplace=True)
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 10)
# print(df.head())

df1 = df.drop(['Population', 'No/100000'], axis=1)
# print(df1.head(50))

A_filter = (df1['Category'] == 'INSTITUTIONS') & (df1['Composition'] == 'TOTAL')
df2 = df1[A_filter]
# print(df2.head(50))
# print(df2.shape)
s = np.arange(348)
df2.set_index(s, inplace=True)
# print(df2.head(50))

A_filter = (df1['Category'] == 'TEACHING STAFF') & (df1['Composition'] == 'TOTAL')
df3 = df1[A_filter]
# print(df3.head(50))
# print(df3.shape)
df3.set_index(s, inplace=True)
# print(df3.head(50))


A_filter = (df1['Category'] == 'ENROLMENT') & (df1['Composition'] == 'TOTAL')
df4 = df1[A_filter]
# print(df4.head(50))
# print(df4.shape)
df4.set_index(s, inplace=True)
# print(df4.head(50))


df5 = df1.drop(['Category', 'Composition', 'Number', 'School Type'], axis=1)
# print(df5.head())

df5['INSTITUTION'] = df2['Number']
df5['TEACHING'] = df3['Number']
df5['ENROLLMENTS'] = df4['Number']

# print(df5.head(23))
# print(df5.shape)
# print(df5['TEACHING'].isna().sum())
df6 = df5.dropna(how='any')
# print(df6.shape)
lr = linear_model.LinearRegression()
X = df6.drop(['District', 'Location', 'TEACHING'], axis=1)
Y = df6['TEACHING']
lr.fit(X, Y)
pickle.dump(lr, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
