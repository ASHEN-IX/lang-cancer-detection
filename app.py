import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load the saved model
model = joblib.load('lung_cancer_model.pkl')

st.title("Lung Cancer Prediction Dashboard (XAI)")
st.write("This dashboard uses an Explainable AI model to predict lung cancer risk based on clinical and lifestyle factors.")

# Sidebar for User Inputs
st.sidebar.header("Patient Data Input")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", (0, 1), help="0: Female, 1: Male")
    age = st.sidebar.slider("Age", 20, 90, 50)
    
    # Mapping for Binary features (1=No, 2=Yes as per dataset structure)
    smoking = st.sidebar.selectbox("Smoking", (1, 2))
    yellow_fingers = st.sidebar.selectbox("Yellow Fingers", (1, 2))
    anxiety = st.sidebar.selectbox("Anxiety", (1, 2))
    peer_pressure = st.sidebar.selectbox("Peer Pressure", (1, 2))
    chronic_disease = st.sidebar.selectbox("Chronic Disease", (1, 2))
    fatigue = st.sidebar.selectbox("Fatigue", (1, 2))
    allergy = st.sidebar.selectbox("Allergy", (1, 2))
    wheezing = st.sidebar.selectbox("Wheezing", (1, 2))
    alcohol = st.sidebar.selectbox("Alcohol Consuming", (1, 2))
    coughing = st.sidebar.selectbox("Coughing", (1, 2))
    shortness_breath = st.sidebar.selectbox("Shortness of Breath", (1, 2))
    swallowing_diff = st.sidebar.selectbox("Swallowing Difficulty", (1, 2))
    chest_pain = st.sidebar.selectbox("Chest Pain", (1, 2))

    data = {
        'GENDER': gender, 'AGE': age, 'SMOKING': smoking, 'YELLOW_FINGERS': yellow_fingers,
        'ANXIETY': anxiety, 'PEER_PRESSURE': peer_pressure, 'CHRONIC DISEASE': chronic_disease,
        'FATIGUE ': fatigue, 'ALLERGY ': allergy, 'WHEEZING': wheezing,
        'ALCOHOL CONSUMING': alcohol, 'COUGHING': coughing, 'SHORTNESS OF BREATH': shortness_breath,
        'SWALLOWING DIFFICULTY': swallowing_diff, 'CHEST PAIN': chest_pain
    }
    return pd.DataFrame(data, index=[0])

df_input = user_input_features()

# Display Input
st.subheader("Patient Profile")
st.write(df_input)

# Prediction
prediction = model.predict(df_input)
prediction_proba = model.predict_proba(df_input)

st.subheader("Prediction Result")
result = "Lung Cancer" if prediction[0] == 1 else "No Lung Cancer"
color = "red" if prediction[0] == 1 else "green"
st.markdown(f"### Result: :{color}[{result}]")
st.write(f"Confidence: **{prediction_proba[0][prediction[0]]*100:.2f}%**")

# XAI Section (SHAP)
st.subheader("Explainable AI (XAI) Interpretation")
st.write("The chart below explains why the model made this specific prediction.")

explainer = shap.TreeExplainer(model)
shap_values = explainer(df_input)

# Plotting the Waterfall
fig, ax = plt.subplots()
shap.plots.waterfall(shap_values[0, :, 1])
st.pyplot(fig)
