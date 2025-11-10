import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# Load saved model and preprocessing objects
rf_model = joblib.load("rf_model2.pkl")
vectorizer = joblib.load("vectorizer2.pkl")
scaler = joblib.load("scaler2.pkl")
label_encoder = joblib.load("label_encoder2.pkl")

st.title("AI Resume Screening App")

# User input fields
experience = st.number_input("Experience (Years)", min_value=0, max_value=50, value=0)
projects = st.number_input("Projects Count", min_value=0, max_value=100, value=0)
ai_score = st.number_input("AI Score (0-100)", min_value=0, max_value=100, value=0)
salary = st.number_input("Salary Expectation ($)", min_value=0, max_value=100000, value=0)
skills = st.text_area("Skills")
education = st.text_area("Education")
certifications = st.text_area("Certifications")
job_role = st.text_area("Job Role")

if st.button("Predict Decision"):
    # Scale numeric features
    num_features = scaler.transform([[experience, projects, ai_score, salary]])
    
    # Combine text features
    text_combined = " ".join([skills, education, certifications, job_role])
    text_features = vectorizer.transform([text_combined])
    
    # Combine numeric + text features
    final_features = hstack([text_features, num_features])
    
    # Predict using the trained model
    pred = rf_model.predict(final_features)
    result = label_encoder.inverse_transform(pred)[0]
    
    st.success(f"Predicted Recruiter Decision: {result}")
