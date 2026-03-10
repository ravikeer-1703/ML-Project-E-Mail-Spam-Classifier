import streamlit as st
import pickle
from preprocessing import text_preprocessing

pipeline = pickle.load(open("pipeline.pkl", "rb"))

title = st.title("Email/SMS Spam Classifier")
st.space()
text_area = st.text_area("Enter your Email here")
button = st.button("Predict")

if button:
    text_process = text_preprocessing(text_area)
    prediction = pipeline.predict([text_process])[0]

    if prediction=="spam":
        st.error("This is Spam")
    else:
        st.success("This is not Spam")