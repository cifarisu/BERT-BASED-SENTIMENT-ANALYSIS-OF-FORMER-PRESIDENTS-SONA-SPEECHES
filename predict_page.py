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
        _, prediction = torch.max(logits, dim=1)
        probability = torch.softmax(logits, dim=1)[0][prediction]

    # Return the predicted sentiment label and score
    return target_names[prediction], probability.item()


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


def show_predict_page():
    st.title("Sentiment Analysis")

    # Get the path of the folder containing the speeches
    folder_path = "./speech_app"

    # Get the names of the files in the folder
    filenames = os.listdir(folder_path)

    # Modify filenames to remove ".csv" and replace underscores with spaces
    file_options = [filename[:-4].replace('_', ' ') for filename in filenames]

    # Add a dropdown to select the file
    selected_filename = st.selectbox("Select a speech", file_options)

    # Add a button to start the sentiment analysis
    if st.button("Analyze Sentiment"):
        # If a file is selected, display the sentiment analysis results
        if selected_filename:
            file_path = os.path.join(folder_path, filenames[file_options.index(selected_filename)])
            groups = sentiment_analysis(file_path)
            fig, ax = plt.subplots()
            ax.plot([g[0] for g in groups], [g[1] for g in groups])
            ax.set_xticks([g[0] for g in groups])
            ax.set_xticklabels([f"Section {g[0]}" for g in groups])
            ax.set_yticks([1, 0, -1])
            uniform_values = [1,0,-1]
            ax.set_yticklabels(uniform_values)
            ax.set_ylabel("Sentiment Score")
            ax.set_xlabel("Section")
            ax.set_title("Sentiment Analysis Results")
            st.pyplot(fig)
            # Display a table with the sentiment scores for each point
            st.write("Sentiment Scores:")
            st.write(pd.DataFrame(groups, columns=["Section", "Score", "Label"]))



if __name__ == '__main__':
    show_predict_page()

