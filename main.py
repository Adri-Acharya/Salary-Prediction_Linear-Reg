import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,root_mean_squared_error

salary=pd.read_csv("Salary Data.csv")

salary.dropna(inplace=True) #2 NaN rows are there
salary=salary.drop_duplicates() #49 duplicates
salary.drop(['Job Title'],axis=1,inplace=True) #contains 150+ categorical column
salary=salary[salary['Salary']>350] # salary 350 is a odd value the immediate next min is 35000
salary=pd.get_dummies(data=salary,columns=['Gender','Education Level'],dtype=bool,drop_first=True) #one hot encoding


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

# def get_user_input():
#     age = int(input("Enter your age: "))
#     year_of_exp = int(input("Year of experience: "))
#     gender = str(input("Enter your Gender (male/female): ")).strip().lower()
#     if gender not in ["male","female"]:
#         raise ValueError ("Enter correct gender")
#     edu_lvl = str(input("Enter your level of education (bachelor's/master's/phd): ")).strip().lower()
#     if edu_lvl not in ["bachelor's","master's","phd"]:
#         raise ValueError ("Enter correct education level")
#     return age,year_of_exp,gender,edu_lvl

# age,year_of_exp,gender,edu_lvl = get_user_input ()
# inp_dict_={
#     "Age": age,
#     "Years of Experience": year_of_exp,
#     "Gender_Male": int(gender == "male"),
#     "Education Level_Master's": int(edu_lvl == "master's"),
#     "Education Level_PhD": int(edu_lvl == "phd")
# }

# df = pd.DataFrame([inp_dict_])[X_test.columns]
# df[['Age','Years of Experience']] = scaler.transform(df[['Age','Years of Experience']])
# predicted_salary = model.predict(df)[0]
# print(f"Predicted salary: {predicted_salary:,.2f}")

import joblib
joblib.dump(model,'model.pkl')
joblib.dump(scaler,'scaler.pkl')
joblib.dump(list(X_train.columns),'columns.pkl')