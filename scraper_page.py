import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup
import pickle

from model import SentimentClassifier
from transformers import BertForSequenceClassification, BertTokenizer

from streamlit.components.v1 import components

# check if CUDA is available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')



import requests
from bs4 import BeautifulSoup
import pickle

# Scrapes transcript data from scrapsfromtheloft.com
def url_to_transcript(url):
    page = requests.get(url).text
    soup = BeautifulSoup(page, "lxml")
    text = [p.text for p in soup.find(class_="entry-content").find_all('p')]
    print(url)
    return text

urls = ['https://www.officialgazette.gov.ph/1998/07/27/joseph-ejercito-estrada-first-state-of-the-nation-address-july-27-1998/',
        'https://www.officialgazette.gov.ph/1999/07/26/joseph-ejercito-estrada-second-state-of-the-nation-address-july-26-1999/',
        'https://www.officialgazette.gov.ph/2000/07/24/joseph-ejercito-estrada-third-state-of-the-nation-address-july-24-2000/',
        'http://www.officialgazette.gov.ph/2001/07/23/gloria-macapagal-arroyo-first-state-of-the-nation-address-july-23-2001/',
        'http://www.officialgazette.gov.ph/2002/07/22/gloria-macapagal-arroyo-second-state-of-the-nation-address-july-22-2002/',
        'http://www.officialgazette.gov.ph/2003/07/28/gloria-macapagal-arroyo-third-state-of-the-nation-address-july-28-2003/',
        'http://www.officialgazette.gov.ph/2004/07/26/gloria-macapagal-arroyo-fourth-state-of-the-nation-address-july-26-2004/',
        'http://www.officialgazette.gov.ph/2005/07/25/gloria-macapagal-arroyo-fifth-state-of-the-nation-address-july-25-2005/',
        'http://www.officialgazette.gov.ph/2006/07/24/gloria-macapagal-arroyo-sixth-state-of-the-nation-address-july-24-2006/',
        'http://www.officialgazette.gov.ph/2007/07/23/gloria-macapagal-arroyo-seventh-state-of-the-nation-address-july-23-2007/',
        'http://www.officialgazette.gov.ph/2008/07/28/gloria-macapagal-arroyo-eighth-state-of-the-nation-address-july-28-2008/',
        'http://www.officialgazette.gov.ph/2009/07/27/gloria-macapagal-arroyo-ninth-state-of-the-nation-address-july-27-2009/',
        'https://www.officialgazette.gov.ph/2010/07/26/state-of-the-nation-address-2010/',
        'https://www.officialgazette.gov.ph/2011/07/25/benigno-s-aquino-iii-second-state-of-the-nation-address-july-25-2011/',
        'https://www.officialgazette.gov.ph/2012/07/23/english-translation-benigno-s-aquino-iii-third-state-of-the-nation-address-july-23-2012/',
        'https://www.officialgazette.gov.ph/2013/07/22/english-benigno-s-aquino-iii-fourth-state-of-the-nation-address-july-22-2013/',
        'https://www.officialgazette.gov.ph/2014/07/28/english-benigno-s-aquino-iii-fifth-state-of-the-nation-address-july-28-2014/',
        'https://www.officialgazette.gov.ph/2015/07/27/english-president-aquino-sixth-sona/',
        'https://www.officialgazette.gov.ph/2016/07/25/rodrigo-roa-duterte-first-state-of-the-nation-address-july-25-2016/',
        'https://www.officialgazette.gov.ph/2017/07/24/rodrigo-roa-duterte-second-state-of-the-nation-address-july-24-2017/',
        'https://www.officialgazette.gov.ph/2018/07/23/rodrigo-roa-duterte-third-state-of-the-nation-address-july-23-2018/',
        'https://www.officialgazette.gov.ph/2019/07/22/rodrigo-roa-duterte-fourth-state-of-the-nation-address-july-22-2019/',
        'https://www.officialgazette.gov.ph/2020/07/27/rodrigo-roa-duterte-fifth-state-of-the-nation-address-july-27-2020/',
        'https://www.officialgazette.gov.ph/2021/07/26/rodrigo-roa-duterte-sixth-state-of-the-nation-address-july-26-2021/']       

# President names
presidents = ['1998 Estrada', '1999 Estrada','2000 Estrada', '2001 Arroyo', '2002 Arroyo', '2003 Arroyo', '2004 Arroyo', '2005 Arroyo', 
             '2006 Arroyo', '2007 Arroyo', '2008 Arroyo', '2009 Arroyo', '2010 Aquino', '2011 Aquino', '2012 Aquino', '2013 Aquino', '2014 Aquino', '2015 Aquino',
             '2016 Duterte', '2017 Duterte', '2018 Duterte', '2019 Duterte', '2020 Duterte', '2021 Duterte']
     
# # Actually request transcripts (takes a few minutes to run)
transcripts = [url_to_transcript(u) for u in urls]


def scrape_transcripts():

    import os

    # Create a directory to hold the text files
    if not os.path.exists("transcripts"):
        os.makedirs("transcripts")

    # Save the transcripts as text files
    for i, c in enumerate(presidents):
        with open("transcripts/" + c + ".txt", "w", encoding="utf-8") as file:
            file.write("\n".join(transcripts[i]))

    # Load the pickled files
    data = {}
    for i, c in enumerate(presidents):
        with open("transcripts/" + c + ".txt", "r", encoding="utf-8") as file:
            data[c] = file.read().splitlines()

    def combine_text(list_of_text):
    #'''Takes a list of text and combines them into one large chunk of text.'''
        combined_text = ' '.join(list_of_text)
        return combined_text

    # Combine it!
    data_combined = {key: [combine_text(value)] for (key, value) in data.items()}


    # We can either keep it in dictionary format or put it into a pandas dataframe
    import pandas as pd
    pd.set_option('max_colwidth',150)

    data_df = pd.DataFrame.from_dict(data_combined).transpose()
    data_df.columns = ['transcript']
    data_df

    import re
    import nltk
    nltk.download('stopwords')
    import string
    import os.path
    from nltk.corpus import stopwords


    def clean_text(text):
        # convert to lowercase
        text = text.lower()
        return text

    # Check if file exists
    stopwords_tl_file = "https://raw.githubusercontent.com/cifcs2/Sentiment-Analysis-of-Former-Presidents-SONA-using-NLP/main/stopwords-tl.txt"

    if os.path.isfile(stopwords_tl_file):
        with open(stopwords_tl_file, "r", encoding="utf8") as f:
            stopwords_tl = set([word.strip() for word in f])
    else:
        print(f"{stopwords_tl_file} not found, using empty set for Filipino stopwords")
        stopwords_tl = set()

    stop_words = set(stopwords.words('english')).union(stopwords_tl)

    def clean_text_round1(text):
        '''Make text lowercase, remove text in square brackets, remove punctuation (except for periods, commas, and exclamation points), remove words containing numbers, remove stopwords, and remove URLs.'''
        text = text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation.replace('.','').replace(',','').replace('!', '')), '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(':', '', text)
        text = ' '.join([word for word in text.split() if word not in stop_words])
        text = re.sub('\s+', ' ', text).strip()
        text = text.strip()
        return text
    
    # Let's take a look at the updated text
    data_clean = pd.DataFrame(data_df.transcript.apply(clean_text_round1))
    data_clean

    # Apply a second round of cleaning
    def clean_text_round2(text):
    #'''Get rid of some additional punctuation, non-sensical text, and tabs that were missed the first time around.'''
        text = re.sub('[‘’“”…]', '', text)
        text = re.sub('\n', '', text)
        text = text.replace('\t', '')
        text = re.sub('state of the nation address', '', text)
        text = re.sub('full transcript', '', text)
        text = re.sub('of his excellency', '', text)
        text = re.sub('ofhis excellency', '', text)
        text = re.sub('president of the philippines during the opening of the regular session of the congress', '', text)
        text = re.sub('during the opening', '', text)
        text = re.sub('of the regular session of the congress', '', text)
        text = re.sub('president of the philippinesto the congress of the philippines', '', text)
        text = re.sub('of her excellency', '', text)
        text = re.sub('president of the philippines', '', text)
        text = re.sub(':', '', text)
        return text

    round2 = lambda x: clean_text_round2(x)

    # Let's take a look at the updated text
    data_clean = pd.DataFrame(data_clean.transcript.apply(clean_text_round2))
    data_clean

    # Let's add the presidents' full names and year as well
    full_namesyears =  ['1998 Estrada', '1999 Estrada','2000 Estrada', '2001 Arroyo', '2002 Arroyo', '2003 Arroyo', '2004 Arroyo', '2005 Arroyo', 
                '2006 Arroyo', '2007 Arroyo', '2008 Arroyo', '2009 Arroyo', '2010 Aquino', '2011 Aquino', '2012 Aquino', '2013 Aquino', '2014 Aquino', '2015 Aquino',
                '2016 Duterte', '2017 Duterte', '2018 Duterte', '2019 Duterte', '2020 Duterte', '2021 Duterte']

    data_clean['full_nameyears'] = full_namesyears
    data_clean['label'] = pd.Series(['speech' for _ in range(len(data_clean.index))])
    data_clean['sentiment_label'] = pd.Series(['speech' for _ in range(len(data_clean.index))])
    data_clean['sentiment_score'] = pd.Series(['speech' for _ in range(len(data_clean.index))])
    data_clean

    import os
    import csv
    import pandas as pd
    import nltk
    nltk.download('punkt')

    from nltk.tokenize import word_tokenize
    from nltk import RegexpParser

    # assuming data_clean contains the DataFrame with transcripts
    sentences = []
    for index, row in data_clean.iterrows():
        president_year = row['full_nameyears']
        label = row['label']
        transcript = row['transcript']
        transcript_sentences = nltk.sent_tokenize(transcript)
        for sentence in transcript_sentences:
            sentences.append([president_year, sentence, label, '', ''])

    sentences_df = pd.DataFrame(sentences, columns=['president_year', 'sentence', 'label', 'sentiment_label', 'sentiment_score'])

    # create a dictionary to store the chunks for each president
    chunks_dict = {}

    # define the chunk size (number of sentences per chunk)
    chunk_size = 10

    # iterate over the sentences in the DataFrame
    for i in range(0, len(sentences_df), chunk_size):
        # get the chunk of sentences
        chunk = sentences_df.iloc[i:i+chunk_size]
        # get the president year for this chunk
        president_year = chunk['president_year'].iloc[0]
        # add the chunk to the dictionary for this president
        if president_year not in chunks_dict:
            chunks_dict[president_year] = []
        chunks_dict[president_year].append(chunk)

    # set the output directory and create it if it does not exist
    output_dir = 'allspeech'
    os.makedirs(output_dir, exist_ok=True)
    
    # write each group of chunks to a separate CSV file
    for president_year, chunks in chunks_dict.items():
        filename = f"{president_year}.csv"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['full_nameyears', 'sentence', 'label', 'sentiment_label', 'sentiment_score'])
            for chunk in chunks:
                for sentence in chunk['sentence'].tolist():
                    writer.writerow([chunk['president_year'].iloc[0], sentence, '', '', ''])

def show_scraper_page():
    st.title("Sentiment Analysis")
    st.markdown("Welcome to the SONA Speech Sentiment Analysis System! 🎙️")
    st.markdown("Explore the sentiments expressed in State of the Nation Address (SONA) speeches delivered by different presidents.")

    if st.button("Get Started"):
        st.info("Retrieving speeches. Please wait...")
        scrape_transcripts()
        st.success("Transcripts successfully retrieved!")

        st.markdown("Let's uncover the sentiments behind the speeches!")
        st.markdown("You can now proceed to the analysis!")

if __name__ == '__main__':
    show_scraper_page()
