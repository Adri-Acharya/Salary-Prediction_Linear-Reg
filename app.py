import pandas as pd
import streamlit as st
import joblib

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')

def get_user_input(age,year_of_exp,gender,edu_lvl):
    age = int(age)
    year_of_exp = int(year_of_exp)
    if year_of_exp >= (age-18):
        raise ValueError("Years of Experience")
    gender = str(gender).strip().lower()
    if gender not in ["male","female"]:
        raise ValueError ("Enter correct gender")
    edu_lvl = str(edu_lvl).strip().lower()
    if edu_lvl not in ["bachelor's","master's","phd"]:
        raise ValueError ("Enter correct education level")
    return age,year_of_exp,gender,edu_lvl

st.title("Salary Predictor")
age = st.slider("Age", min_value=18, max_value=70)
year_of_exp =st.number_input("Years of Experience",min_value = 0, max_value=50)

gender = st.radio("Gender",options=['Male','Female'])
edu_lvl = st.selectbox("Education level",options=["Bachelor's","Master's","PhD"])

if st.button("Predict"):
    age,year_of_exp,gender,edu_lvl = get_user_input (age,year_of_exp,gender,edu_lvl)
    inp_dict_={
        "Age": age,
        "Years of Experience": year_of_exp,
        "Gender_Male": int(gender == "male"),
        "Education Level_Master's": int(edu_lvl == "master's"),
        "Education Level_PhD": int(edu_lvl == "phd")
    }

    df = pd.DataFrame([inp_dict_])[columns]
    df[['Age','Years of Experience']] = scaler.transform(df[['Age','Years of Experience']])
    predicted_salary = model.predict(df)[0]
    
    st.write(f"Predicted salary: {predicted_salary:,.2f} rs")