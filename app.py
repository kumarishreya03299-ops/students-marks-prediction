import streamlit as st
import joblib

st.title("🎓 Student Marks Prediction")
st.write("Enter student details to predict marks.")

number_courses = st.number_input(
    "Number of Courses",
    min_value=1,
    max_value=10,
    value=3
)

time_study = st.number_input(
    "Study Time (hours)",
    min_value=0.0,
    max_value=24.0,
    value=5.0
)

model = joblib.load("model.pkl")

if st.button("Predict Marks"):
    prediction = model.predict([[number_courses, time_study]])

    st.success(f"Predicted Marks: {prediction[0]:.2f}")