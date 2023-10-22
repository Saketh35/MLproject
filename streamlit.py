import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Create a function to get user input and make predictions
def predict_datapoint():
    st.title("Student Exam Performance Indicator")
    st.write("## Student Exam Performance Prediction")

    gender = st.selectbox("Gender", ["Male", "Female"])
    ethnicity = st.selectbox("Race Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
    parental_level_of_education = st.selectbox("Parental Level of Education", ["associate's degree", "bachelor's degree", "high school", "master's degree", "some college", "some high school"])
    lunch = st.selectbox("Lunch Type", ["free/reduced", "standard"])
    test_preparation_course = st.selectbox("Test Preparation Course", ["none", "completed"])
    reading_score = st.slider("Reading Score out of 100", 0, 100, 50)
    writing_score = st.slider("Writing Score out of 100", 0, 100, 50)

    if st.button("Predict"):
        data = CustomData(
            gender="male" if gender == "Male" else "female",
            race_ethnicity=ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course.lower(),
            reading_score=float(reading_score),
            writing_score=float(writing_score)
        )
        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)
        result = round(float(result), 2)
        st.success(f"The predicted score is: {result}")

# Main function to run the Streamlit app
def main():
    st.header("Welcome to the home page")
    st.write("This is a simple web application for predicting student exam performance.")
    predict_datapoint()

# Run the Streamlit app
if __name__ == "__main__":
    main()
