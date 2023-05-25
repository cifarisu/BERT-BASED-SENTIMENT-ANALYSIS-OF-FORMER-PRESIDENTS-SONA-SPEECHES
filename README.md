# BERT-BASED-SENTIMENT-ANALYSIS-OF-FORMER-PRESIDENTS-SONA-SPEECHES

A system for sentiment analysis of State of the Nation Address (SONA) speeches given by former presidents. The system utilizes the BERT (Bidirectional Encoder Representations from Transformers) model for sentiment classification.

# Contributors
* Mark Jerome Cifra
* Lenard Corporal
* Allen Marchel
* Jessa Lorenza Sabas


About
This project was developed as a partial completion for Software Engineering 2 (CS 116) at Bicol University. The system aims to analyze the sentiment expressed in the speeches of the four previous presidents of the Philippines: Estrada, Arroyo, Aquino, and Duterte.

# Features
Data scraping: Scrapes official gazette speeches from 1935 to 1997 for training the model.

Data cleaning and preprocessing: Prepares the scraped data by applying cleaning and preprocessing techniques.

Sentence chunking: Segments the speeches into single sentences for analysis.

Annotation: Utilizes the Hugging Face BERT uncased model pipeline for automatic annotation of the training dataset.

Model training: Fine-tunes the BERT model using the annotated dataset for sentiment analysis.

Streamlit app system: Provides a user interface where users can select speeches, view sentiment analysis results, and visualize sentiment trends using line graphs.

Data scraping for recent speeches: Incorporates a data scraper to retrieve speeches from 1998 to 2021, focusing on the four previous presidents.

Line graph visualization: Divides the sentiment analysis results into 10 sections for better comprehension of the overall speech sentiment.
