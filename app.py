import streamlit as st
import pandas as pd
import pickle

import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


ps = PorterStemmer()


# 1. preprocess
# 2. vectorise
# 3. Predict
# 4. Display the output

# stemming and apply all the steps

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)





tfidf = pickle.load(open('vectorizer1.pkl', 'rb'))
model  = pickle.load(open('model1.pkl', 'rb'))

st.title('Email/SMS Spam Classifier')
input_sms = st.text_area("Enter the message ")

if st.button('Predict'):



    # vectorise
    transfrmed_sms = transform_text(input_sms)

    # vectorise
    vector_input = tfidf.transform([transfrmed_sms])

    # prediction
    result = model.predict(vector_input)[0]

    # display
    if result ==1:
        st.header("Spam")
    else:
        st.header("Not Spam")
