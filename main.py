import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
from tensorflow.keras.models import load_model

### Mapping of words index back to words(for understanding)
word_index = imdb.get_word_index()

reverse_word_idx = { value:key for key,value in word_index.items()}

model = load_model('simple_rnn_imdb.h5')

## Helper Function for the decoding review
def decode_review(encoded_review):
    return ' '.join([reverse_word_idx.get(i-3,'?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review


## Step 3 :

## Prrediction

def predict_sentiment(review):
    preprocess= preprocess_text(review)
    pred = model.predict(preprocess)

    sentiment = "Positive" if pred[0][0] > 0.5 else "Negative"
    
    return sentiment,pred[0][0]


## Streamlit
import streamlit as st
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative")

user_input = st.text_area("Movie Review")
if st.button("Classify"):
    inp = preprocess_text(user_input)
    
    pred = model.predict(inp)
    
    sentiment = "Positive" if pred[0][0] > 0.5 else "Negative"
    st.write("Sentiment",sentiment)
    st.write("Prediction Score:",pred[0][0])
else:
    st.write("Please enter a movie review ")
    

    