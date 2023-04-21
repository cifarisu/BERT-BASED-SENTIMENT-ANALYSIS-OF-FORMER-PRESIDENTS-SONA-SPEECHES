import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

from model import SentimentClassifier
from transformers import BertForSequenceClassification, BertTokenizer

# check if CUDA is available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = BertForSequenceClassification.from_pretrained('bert-base-cased')
model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')), strict=False)
model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

target_names = ['NEGATIVE', 'POSITIVE']
MAX_LEN = 160

# Assuming you have already defined `tokenizer`, `model`, `device`, and `target_names`
# `tokenizer` and `model` should be the ones you used for training

def predict_sentiment(text):
    # Tokenize the text and convert it to input tensors for the model
    encoded_text = tokenizer.encode_plus(
        text,
        max_length=MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)

    # Pass the input tensors to the model to get the predicted sentiment
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        logits = output.logits
        probabilities = torch.softmax(logits, dim=1).squeeze().tolist()

    # Get the predicted sentiment label and score
    predicted_label = target_names[np.argmax(probabilities)]
    confidence = probabilities[np.argmax(probabilities)]

    # Return the predicted sentiment label, confidence, and probabilities
    return {'sentiment': predicted_label, 'confidence': confidence, 'probabilities': {
        'negative': probabilities[0],
        'positive': probabilities[1]
    }}


def sentiment_analysis(file):
    # Read the file into a Pandas dataframe
    df = pd.read_csv(file)

    # Create new columns for sentiment label and score
    df['sentiment_label'] = ''
    df['sentiment_score'] = 0.0

    # Loop through each row in the dataframe and predict the sentiment label and score for each sentence
    for i, row in df.iterrows():
        sentence = row['sentence']
        sentiment_label, sentiment_score = predict_sentiment(sentence)
        
        if sentiment_label == "NEGATIVE":
            sentiment_score = -sentiment_score
        df.at[i, 'sentiment_label'] = sentiment_label
        df.at[i, 'sentiment_score'] = sentiment_score

    # Group the rows into 10 equal parts and calculate the average sentiment score for each group
    num_groups = 10
    group_size = len(df) // num_groups
    groups = []
    for i in range(num_groups):
        start = i * group_size
        end = start + group_size
        group_df = df.iloc[start:end]
        avg_score = group_df['sentiment_score'].mean()
        groups.append((i+1, avg_score, group_df['sentiment_label'].tolist()))

    return groups


def show_predict_pages():
    st.title("Sentiment Analysis")

    # Add a text input to enter a sentence to analyze
    sentence = st.text_input("Enter a sentence to analyze:")

    # Add a button to start the sentiment analysis
    if st.button("Analyze Sentiment", key='analyze_sentiment_button'):
        # If a sentence is entered, display the sentiment analysis results
        if sentence:
            result = predict_sentiment(sentence)
            st.write(result)

if __name__ == '__main__':
    show_predict_pages()

