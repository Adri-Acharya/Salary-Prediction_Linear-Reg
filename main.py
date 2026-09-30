import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,root_mean_squared_error
from sklearn.metrics import mean_absolute_error

salary=pd.read_csv("Salary Data.csv")

salary.dropna(inplace=True) #2 NaN rows are there
salary=salary.drop_duplicates() #49 duplicates

t = salary['Job Title'].str.lower()

senior = t.str.contains(r'\b(?:senior|director|chief|vp|principal|lead|head)\b|\b(?:ceo|cto|cfo)\b')
junior = t.str.contains(r'\b(?:junior|entry|intern|associate|assistant|trainee)\b')
salary['Level'] = np.select([senior, junior], ['senior', 'junior'], default='mid')
salary.drop(columns=['Job Title'], inplace=True)

salary=salary[salary['Salary']>350] # salary 350 is possible typo as the next  immediate min is 35000
salary=pd.get_dummies(data=salary,columns=['Gender','Education Level','Level'],dtype=bool,drop_first=True) #one hot encoding


X= salary.drop(columns=['Salary'])
y= salary['Salary']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state= 2)

scaler = StandardScaler()

X_train[['Age','Years of Experience']] = scaler.fit_transform(X_train[['Age','Years of Experience']])   # fit AND transform on train
X_test[['Age','Years of Experience']] = scaler.transform(X_test[['Age','Years of Experience']])

model=LinearRegression()
model.fit(X_train,y_train)

y_test_predicted=model.predict(X_test)
y_train_predicted=model.predict(X_train)

print(f"Train R2 Score: {r2_score(y_train, y_train_predicted):.4f}")
print(f"Test R2 Score: {r2_score(y_test, y_test_predicted):.4f}")
print(f"Train RMSE: {root_mean_squared_error(y_train, y_train_predicted):.2f}")
print(f"Test RMSE: {root_mean_squared_error(y_test, y_test_predicted):.2f}")

print(f"Test MAE: {mean_absolute_error(y_test, y_test_predicted):.2f}")
print(f"Train MAE: {mean_absolute_error(y_train, y_train_predicted):.2f}")

import joblib
joblib.dump(model,'model.pkl')
joblib.dump(scaler,'scaler.pkl')
joblib.dump(list(X_train.columns),'columns.pkl')