import nltk
import pickle
import re
import string
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import  WordNetLemmatizer
wnl = WordNetLemmatizer()
stopwords_list = set(stopwords.words("english"))
nltk.download("stopwords")
nltk.download("punkt")


Words_dict = pickle.load(open("words_vocab.pkl", "rb"))

# Text preprocessing function:
def text_preprocessing(text):
    text = text.lower()
    # Convert short words chat words to full words
    final_text = []
    for words in re.split(r"[ ,';)(|.]+", text):
        if words in Words_dict:
            final_text.append(Words_dict[words])
        else:
            final_text.append(words)

    # English Contraction words converter like i'll to i will
    text = final_text[:]
    final_text.clear()
    for words in text:
        if words in Words_dict:
            final_text.append(Words_dict[words])
        else:
            final_text.append(words)

    # Replacing what's app chats words with english meaning words
    text = final_text[:]
    final_text.clear()
    for words in text:
        if words in Words_dict:
            final_text.append(Words_dict[words])
        else:
            final_text.append(words)

    # Removing Digits and punctuations
    text = " ".join(final_text)
    final_text.clear()
   # text = re.sub(r"\d+", "", str(text))        # remove digits
    text = text.translate(str.maketrans("" ,"" ,string.punctuation)) # punctuations

    # Replacing suffix and prefix word into full word
    for words in text.split():
        if words in Words_dict:
            final_text.append(Words_dict[words])
        else:
            final_text.append(words)

    # Removing words other than english words
    text = final_text[:]
    final_text.clear()
    for words in text:
        if words.isascii():
            final_text.append(words)
        else:
            final_text.append("")

    # Removing stopwords
    text = final_text[:]
    final_text.clear()
    for words in text:
        if words in stopwords_list or len(words )<2:
            final_text.append("")
        else:
            final_text.append(words)

    # Lemmatization: converting word to their baseform word
    text = final_text[:]
    final_text.clear()
    for word in text:
        v_text = wnl.lemmatize(word, pos="v")   # transform verb: like don't to do not
        n_text = wnl.lemmatize(v_text, pos="n") # transform noun: like cars to car
        a_text = wnl.lemmatize(n_text, pos="a") # transform adjective: like better to good
        final_text.append(a_text)

    # Removing words that have only one character
    text = final_text[:]
    final_text.clear()
    for words in text:
        if len(words)>=2:
            final_text.append(words)
        else:
            final_text.append("")

    final_text = re.sub(r"\s+", " ", " ".join(final_text)) # Removing extra spaces
    return final_text