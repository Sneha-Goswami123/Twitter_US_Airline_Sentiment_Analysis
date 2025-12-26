"""# ================================
# Module 2: Text Preprocessing
# Twitter US Airline Sentiment
# ================================"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (only first time)
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("data/Tweets.csv")

# Keep only required columns
df = df[['text', 'airline_sentiment']]

# Load English stopwords
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()                       # lowercase
    text = re.sub(r"http\S+", "", text)       # remove URLs
    text = re.sub(r"@\w+", "", text)          # remove mentions
    text = re.sub(r"[^a-z\s]", "", text)      # remove punctuation & numbers
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Apply preprocessing
df['clean_text'] = df['text'].apply(clean_text)

# Show before vs after
print("ORIGINAL TWEET:\n", df['text'].iloc[0])
print("\nCLEANED TWEET:\n", df['clean_text'].iloc[0])

# Save cleaned data for Module 3
df.to_csv("data/cleaned_tweets.csv", index=False)

print("\nPreprocessing complete.")
print("Cleaned dataset saved as data/cleaned_tweets.csv")
