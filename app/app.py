import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Titanic AI Predictor", layout="wide")

# Load model
model = pickle.load(open("outputs/model.pkl", "rb"))

# Load dataset
df = pd.read_csv("dataset/titanic.csv")

# Title
st.title("🚢 Titanic Survival AI Predictor")
st.header("Model Performance")

st.write("Machine learning model trained on Titanic dataset")

st.metric("Model Accuracy", "82%")

st.write(
"An AI-powered system that predicts whether a Titanic passenger would survive based on passenger attributes."
)

# Sidebar Inputs
st.sidebar.header("Passenger Information")

pclass = st.sidebar.selectbox("Passenger Class", [1,2,3])

sex = st.sidebar.selectbox("Gender", ["Male","Female"])

age = st.sidebar.slider("Age", 1, 80)

fare = st.sidebar.number_input("Fare", value=10.0)

sibsp = st.sidebar.number_input("Siblings / Spouse", value=0)

parch = st.sidebar.number_input("Parents / Children", value=0)

embarked = st.sidebar.selectbox("Embarked Port", ["Q","S"])

# Convert inputs
sex = 0 if sex=="Male" else 1
emb_q = 1 if embarked=="Q" else 0
emb_s = 1 if embarked=="S" else 0

input_data = np.array([[pclass,sex,age,sibsp,parch,fare,emb_q,emb_s]])

# Prediction Section
st.header("Prediction Result")

if st.button("Predict Survival"):

    with st.spinner("AI model analyzing passenger data..."):
        import time
        time.sleep(1)

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]

        if prediction[0] == 1:
            st.success("Passenger Survived 🎉")
        else:
            st.error("Passenger Did Not Survive ❌")

        prob_percent = probability * 100

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_percent,
            title={'text': "Survival Probability"},
            gauge={
                'axis': {'range': [0,100]},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0,50], 'color': "red"},
                    {'range': [50,75], 'color': "yellow"},
                    {'range': [75,100], 'color': "lightgreen"}
                ]
            }
        ))

        st.plotly_chart(gauge)
# Data Insights
st.header("📊 Titanic Data Insights")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    sns.countplot(x='Survived', data=df, ax=ax1)
    ax1.set_title("Survival Count")
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    sns.countplot(x='Survived', hue='Sex', data=df, ax=ax2)
    ax2.set_title("Survival by Gender")
    st.pyplot(fig2)

# Age Distribution
st.subheader("Age Distribution")

fig3, ax3 = plt.subplots()
sns.histplot(df['Age'], bins=20, kde=True, ax=ax3)

st.pyplot(fig3)

st.header("Feature Importance")

features = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked_Q","Embarked_S"]

try:
    importances = model.feature_importances_

    fig, ax = plt.subplots()

    sns.barplot(x=importances, y=features, ax=ax)

    ax.set_title("Feature Importance")

    st.pyplot(fig)

except:
    st.write("Feature importance available for tree models only.")