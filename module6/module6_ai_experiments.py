# ==========================================
# Module 6: AI Exploration Experiments
# (No NLTK required)
# ==========================================

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Simple text cleaning (lightweight)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

# Load dataset
df = pd.read_csv("data/cleaned_tweets.csv")

df['clean_text'] = df['clean_text'].fillna("")
df = df[df['clean_text'].str.strip() != ""]

X = df['clean_text']
y = df['airline_sentiment']

# Train model (same as previous modules)
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_tfidf, y)

# -------------------------------
# AI Exploration Experiments
# -------------------------------
test_inputs = [
    "This flight was absolutely amazing!!! 😍",
    "Worst airline experience ever.",
    "The flight is scheduled at 5 PM.",
    "I love the service but hate the delay",
    "@United 😡😡😡",
    "Okay experience, nothing special."
]

print("\nAI Exploration Results:\n")

for text in test_inputs:
    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]

    print(f"Input Text       : {text}")
    print(f"Cleaned Text     : {cleaned}")
    print(f"Predicted Output : {prediction}")
    print("-" * 40)
