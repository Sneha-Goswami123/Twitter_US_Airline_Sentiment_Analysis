"""# ==========================================
# Module 5: Real-Time Sentiment Analysis App
# =========================================="""

import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download stopwords (first time only)
nltk.download('stopwords')

# ---------- Text cleaning ----------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ---------- Load & train model ----------
df = pd.read_csv("data/cleaned_tweets.csv")
df['clean_text'] = df['clean_text'].fillna("")
df = df[df['clean_text'].str.strip() != ""]

X = df['clean_text']
y = df['airline_sentiment']

tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_tfidf, y)

# ---------- Streamlit UI ----------
st.title("✈️ Twitter Airline Sentiment Analyzer")
st.write("Enter a tweet below to analyze its sentiment in real time.")

user_input = st.text_area("Enter Tweet Text:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == "positive":
            st.success("😊 Positive Sentiment")
        elif prediction == "neutral":
            st.info("😐 Neutral Sentiment")
        else:
            st.error("😡 Negative Sentiment")
