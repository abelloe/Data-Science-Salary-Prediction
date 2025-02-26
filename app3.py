import streamlit as st
import numpy as np
import joblib

model = joblib.load('model.pkl')

st.set_page_config(page_title="Different Data Science Roles Salary Prediction", page_icon="💼", layout="wide")
st.title("💼 Data Science Salary Predictor")
st.markdown("Predict salaries for various data science roles based on your inputs. Get insights and plan your career effectively!")

st.sidebar.header("Kindly provide your job details:")
experience = st.sidebar.selectbox("🧑‍💻 What's your level of experience", ['Senior level', 'Mid-level', 'Entry level', 'Experienced level'])
employment = st.sidebar.selectbox("📋 Your type of employment", ['Full time', 'Part time', 'Contract', 'Freelance'])
company = st.sidebar.selectbox("🏢 What is your company size", ['Large', 'Middle', 'Small'])
job = st.sidebar.selectbox("💻 Choose your job title", ['Data Scientist', 'Data Analyst', 'Data Engineer', 'Data Architect'])

experience_mapping = {'Entry level': 0, 'Experienced level': 1, 'Mid-level': 2, 'Senior level': 3}
employment_mapping = {'Contract': 0, 'Freelance': 1, 'Full time': 2, 'Part time': 3}
company_mapping = {'Large': 0, 'Middle': 1, 'Small': 2}
job_mapping = {'Data Analyst': 0, 'Data Architect': 1, 'Data Engineer': 2, 'Data Scientist': 3}

experience_level = experience_mapping[experience]
employment_type = employment_mapping[employment]
company_size = company_mapping[company]
job_title = job_mapping[job]


features = np.array([[experience_level, employment_type, company_size, job_title]])
padded_input = np.zeros(8)
#padded_input[:4] = features
padded_input[1] = features[0,0]
padded_input[2] = features[0,1]
padded_input[7] = features[0,2]
padded_input[3] = features[0,3]
padded_input = padded_input.reshape(1, -1)

st.subheader("Salary Prediction")
if st.button("💰 Predict Salary"):
    prediction = model.predict(padded_input)
    st.success(f"🤑 Predicted Salary: **${prediction[0]:,.2f}**")
    st.balloons()

# Add Additional Information Section
st.markdown("---")
st.subheader("💡 About This App")
st.write("""
This app predicts salaries for various data science roles using machine learning. Simply provide job details in the sidebar to get an estimated salary range. 
It is designed to assist professionals and organizations in making data-driven career and hiring decisions.
""")

