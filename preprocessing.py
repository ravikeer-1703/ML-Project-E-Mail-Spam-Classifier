import pickle
import re

Words_dict = pickle.load(open("words_vocabulary.pkl", "rb"))

from nltk.stem import  WordNetLemmatizer
wnl = WordNetLemmatizer()

# Text preprocessing function:
def text_preprocessing(text):
    text = text.lower()
    tokens = text.split()

    process_words = []
    for token in tokens:
        # Handle contractions
        if token in Words_dict:
            process_words.append(Words_dict[token])
        else:
            process_words.append(token)

        # Lemmatization
    process_words = " ".join(process_words)
    token1 = wnl.lemmatize(process_words, pos="v")
    token2 = wnl.lemmatize(token1, pos="n")
    token3 = wnl.lemmatize(token2, pos="a")
    digit_remove = re.sub(r"\d+", "", token3)

    # handling repeating words by allowing only 2 time repeatation
    final_output = re.sub(r"(.)\1{3,}", r"\1\1", digit_remove)
    final_text = re.sub(r"\s+", " ", final_output)
    return final_text.strip()