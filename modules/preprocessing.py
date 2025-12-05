import re
import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

stop_factory = StopWordRemoverFactory()
stopword_sastrawi = stop_factory.create_stop_word_remover()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Case Folding
    text = text.lower()
    
    # 2. Hapus Mention, Link, Hashtag
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    
    # 3. Hapus Angka dan Tanda Baca
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # 4. Normalisasi Whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_dataframe(df, text_col='content'):
    # Simpan teks asli
    df['clean_content'] = df[text_col].apply(clean_text)
    
    # Stopword Removal (Sastrawi)
    df['clean_content'] = df['clean_content'].apply(lambda x: stopword_sastrawi.remove(x))
    
    # Hapus data kosong setelah cleaning
    df = df[df['clean_content'] != '']
    return df

def map_score_to_label(score):
    if score <= 2:
        return 0 # Negatif
    elif score == 3:
        return 1 # Netral
    else:
        return 2 # Positif

def get_label_name(label):
    if label == 0: 
      return "Negatif"
    elif label == 1: 
      return "Netral"
    else: 
      return "Positif"