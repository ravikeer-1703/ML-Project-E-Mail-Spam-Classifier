import streamlit as st
import pickle
from preprocessing import text_preprocessing
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
nltk.download('stopwords')

pipeline = pickle.load(open("pipeline.pkl", "rb"))

title = st.title("Email/SMS Spam Classifier")
st.space()
text_area = st.text_area("Enter your Email here")
button1 = st.button("Predict", key=0)

if text_area:
    if button1:
        text_process = text_preprocessing(text_area)
        prediction = pipeline.predict([text_process])[0]

        if prediction=="spam":
            st.error("This is Spam")
        elif prediction=="ham":
            st.success("This is not Spam")
        else:
            st.success("")

    st.space()
    input_static = st.text("Statics of Your input")
    col1, col2, col3 = st.columns(3)
    col1.metric("Characters", len(text_area))
    col2.metric("Words", len(re.split(r"[ .!)('/]+", text_area)))
    col3.metric("Sentences", len(re.split(r"[.]+", text_area)))

st.space()
st.text("------------------------------------------------------------------------------------------------------------------------------------------")

st.subheader("Or Upload CSV File Here")
csv_file = st.file_uploader("Choose a file", type = ["csv"])
button2 = st.button("Predict", key=1)

if csv_file is not None:
    df = pd.read_csv(csv_file)
    st.write("Dataset Preview")
    st.dataframe(df.head())
    data_column = st.selectbox("Select Column", df.columns)

    if button2:
        text_process = df[data_column].apply(text_preprocessing)
        prediction = pipeline.predict(text_process)
        spam_mail = []
        ham_mail = []

        for label in prediction:
            if label=="spam" or label==1:
                spam_mail.append(label)
            else:
                ham_mail.append(label)


        st.space()
        input_static = st.text("Statics of Your input")
        col4, col5, col6 = st.columns(3)
        col4.metric("Number of Spam mails", len(spam_mail))
        col5.metric("Number of Ham mails", len(ham_mail))
        col6.metric("Percentage of Spam mails", np.round((len(spam_mail)/len(ham_mail))*100,2))

        st.space()
        x = ["spam", "ham"]
        y = [len(spam_mail), len(ham_mail)]
        fig, ax = plt.subplots()
        ax.bar(x, y, color = ["red", "green"])
        ax.set_facecolor("lightgrey")
        fig.set_facecolor("lightgrey")
        ax.tick_params(axis="x", colors="black")
        ax.tick_params(axis="y", colors="black")
        st.pyplot(fig)