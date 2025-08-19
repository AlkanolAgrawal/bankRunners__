import streamlit as st 
import numpy as np 
import os
import tensorflow as tf 
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler


import os
import tensorflow as tf
import pickle

# Model
model = tf.keras.models.load_model(os.path.join("artifacts", "ann_model.h5"))

# Scaler
with open(os.path.join("artifacts", "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

# OneHotEncoder
with open(os.path.join("artifacts", "OneHotEncoder_Geography.pkl"), "rb") as f:
    ohe_geo = pickle.load(f)

# LabelEncoder
with open(os.path.join("artifacts", "Label_Encoder_Gender.pkl"), "rb") as f:
    lab_enco_gen = pickle.load(f)



st.title("Bank Customer Churn Prediction")
geography = st.selectbox("Select Geography", ohe_geo.categories_[0])
gender = st.selectbox("Select Gender", lab_enco_gen.classes_)
age = st.number_input("Enter Age", min_value=0, max_value=100, value=30)
balance = st.number_input("Enter Balance", min_value=0.0, value=0.0)
num_of_products = st.number_input("Enter Number of Products", min_value=1, max_value=5, value=1)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Enter Estimated Salary", min_value=0.0, value=0.0)
tenure = st.number_input("Enter Tenure (in years)", min_value=0, value=10)
credit_score = st.number_input("Enter Credit Score", min_value=0, max_value=1000, value=600)

input_data = {
    # 'Geography': geography,
    'CreditScore': credit_score,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary,
}
input_df = pd.DataFrame([input_data])
geo_encoded = ohe_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out())
input_df = pd.concat([input_df, geo_encoded_df], axis=1)
input_df['Gender'] = lab_enco_gen.transform(input_df['Gender'])
# print(input_df)

scaled_data = scaler.transform(input_df)
prediction = model.predict(scaled_data)
prob=prediction[0][0]
if prob > 0.5:
    st.write(f"The customer is likely going to churn from the bank, with a prediction prob of {prob:.4f}")
else:
    st.write(f"The customer is likely not going to churn from the bank, with a prediction prob of {prob:.4f}")