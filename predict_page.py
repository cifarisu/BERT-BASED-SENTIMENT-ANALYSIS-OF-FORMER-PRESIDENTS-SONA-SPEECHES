import streamlit as st
import numpy as np
import torch
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

    
    # Return the predicted sentiment label
    return target_names[prediction]

def show_predict_page():
    st.title("Sentiment Analysis")

    # Add a text input field to the app
    text_input = st.text_input("Enter a text", "")

    # Add a button to trigger the prediction
    if st.button("Predict"):
        # Get the predicted sentiment label
        sentiment = predict_sentiment(text_input)
        
        # Display the predicted sentiment
        st.write(f"Predicted sentiment: {sentiment}")

if __name__ == '__main__':
    show_predict_page()
