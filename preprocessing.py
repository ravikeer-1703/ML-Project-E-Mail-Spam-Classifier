import nltk
import pickle
import re
import string
import streamlit as st

@st.cache_resource
def down_nltk():
    nltk.download("stopwords")
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("stopwords")
down_nltk()

Words_dict = pickle.load(open("words_vocab.pkl", "rb"))

# Text preprocessing function:
from nltk.corpus import stopwords
from nltk.stem import  WordNetLemmatizer

wnl = WordNetLemmatizer()
stopwords_list = set(stopwords.words("english"))

def text_preprocessing(text):
    # Clean text
    text = re.sub(r"\d+", "", str(text))
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()

    # Tokenize and process
    tokens = text.split()
    processed_tokens = []

    for token in tokens:
        # Keep only ASCII words
        if not token.isascii():
            continue

        # Remove stopwords
        if token in stopwords_list:
            continue

        # Handle contractions
        if token in Words_dict:
            token = Words_dict[token]

        # Lemmatization
        token = wnl.lemmatize(token, pos="v")
        token = wnl.lemmatize(token, pos="n")
        token = wnl.lemmatize(token, pos="a")

        # Keep words with length >= 2
        if len(token) > 2:
            processed_tokens.append(token)

    # Join and clean
    final_output = " ".join(processed_tokens)
    final_output = re.sub(r"\s+", " ", final_output)
    final_output = re.sub(r"(.)\1{3,}", r"\1\1", final_output)

    return final_output.strip()