import re
import pandas as pd
import os

KAMUS_MANUAL = {
    'fc': 'berhenti paksa','force close': 'berhenti paksa',
    'out': 'keluar','log out': 'keluar','sign out': 'keluar',
    'apk': 'aplikasi','apps': 'aplikasi','aps': 'aplikasi','app': 'aplikasi',
    'dev': 'pengembang','developer': 'pengembang','admin': 'administrator',
    'log in': 'masuk','login': 'masuk','sign up': 'daftar','sign in': 'masuk',
    'ngelag': 'lambat','lag': 'lambat','lemot': 'lambat','lola': 'lambat','lelet': 'lambat',
    'bapuk': 'buruk','burig': 'burik',
    'bug': 'kesalahan sistem','error': 'kesalahan',
    'typo': 'salah ketik',
    'donlot': 'unduh','download': 'unduh',
    'install': 'pasang','instal': 'pasang',
    'uninstal': 'hapus','uninstall': 'hapus',
    'bintang': 'nilai','rate': 'nilai','rating': 'nilai',
    'gk': 'tidak', 'gak': 'tidak', 'ga': 'tidak', 'nggak': 'tidak',
    'bgs': 'bagus', 'keren': 'bagus', 'mantap': 'bagus',
    'jelek': 'buruk', 'parah': 'buruk',
    'sy': 'saya', 'gw': 'saya', 'aku': 'saya',
    'sdh': 'sudah', 'udh': 'sudah', 'dah': 'sudah',
    'blm': 'belum',
    'krn': 'karena', 'karna': 'karena',
    'tp': 'tapi',
    'tpi': 'tapi',
    'yg': 'yang',
    'dgn': 'dengan',
    'utk': 'untuk',
    'gabisa': 'tidak bisa', 'gbisa': 'tidak bisa',
    'gasuka': 'tidak suka',
    'kalo': 'kalau', 'klo': 'kalau',
    'bgt': 'banget', 'dlm': 'dalam',
    'tdk': 'tidak', 'jg': 'juga',
    'pake': 'pakai', 'n': 'dan', '&': 'dan'
}

SLANG_DICT = {}

def load_slang_dictionary(csv_path='assets/colloquial-indonesian-lexicon.csv'):
    global SLANG_DICT
    if SLANG_DICT: return SLANG_DICT
    if os.path.exists(csv_path):
        try:
            df_kamus = pd.read_csv(csv_path, encoding='latin-1', header=None)
            slang_dict_csv = dict(zip(df_kamus[0], df_kamus[1]))
            SLANG_DICT.update(slang_dict_csv)
            print(f"Berhasil memuat {len(slang_dict_csv)} kata dari CSV.")
        except Exception as e:
            print(f"Gagal memuat CSV: {e}")
        
    SLANG_DICT.update(KAMUS_MANUAL)
    return SLANG_DICT

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'@[A-Za-z0-9_]+', '', text) # Hapus Mention
    text = re.sub(r'#\w+', '', text)           # Hapus Hashtag
    text = re.sub(r'http\S+', '', text)        # Hapus URL
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)   # Hapus Angka & Tanda Baca
    text = re.sub(r'\s+', ' ', text).strip()   # Hapus Spasi Berlebih
    return text

def normalize_slang(text):
    if not SLANG_DICT: load_slang_dictionary()
    words = text.split()
    normalized_words = [SLANG_DICT.get(word, word) for word in words]
    return " ".join(normalized_words)

def preprocess_dataframe(df, text_col='content'):
    load_slang_dictionary()
    # 1. Cleaning
    df['clean_content'] = df[text_col].apply(clean_text)
    # 2. Normalization
    df['clean_content'] = df['clean_content'].apply(normalize_slang)
    # 3. Filter kosong
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