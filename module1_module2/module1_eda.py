"""
Real Time Sentiment Analysis System
Module 1: Data Exploration & Analysis

Project: Twitter US Airline Sentiment Analysis
Data Modality: Textual Data (Tweets)
Dataset: Twitter US Airline Sentiment Dataset (Kaggle)

Made By:
    Sneha   | 102303723 | 3C52

Date: December 2025
"""

# Import Libraries
import matplotlib
matplotlib.use('Agg')   # non-GUI backend

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("data/Tweets.csv")

# Basic dataset Inspection
print(df.head())
print(df.info())

# Sentiment Distribution
print(df['airline_sentiment'].value_counts())

df['airline_sentiment'].value_counts().plot(
    kind='bar', title='Sentiment Distribution'
)
plt.savefig("sentiment_distribution.png")
plt.close()


# Sample Tweets
print("\nSample Tweets:\n")
print(df['text'].head(5))

# Tweet Length Analysis
df['tweet_length'] = df['text'].apply(len)

df['tweet_length'].plot(
    kind='hist', bins=30, title='Tweet Length Distribution'
)
plt.xlabel("Tweet Length")
plt.show()

"""Dataset contains positive, negative, and neutral tweets
Used for sentiment classification in social media domain"""
