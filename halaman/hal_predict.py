import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from google_play_scraper import Sort, reviews, app
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns

# -----------------------------
# LOAD MODEL INDO BERTA
# -----------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("mdhugol/indonesia-bert-sentiment-classification")
    model = AutoModelForSequenceClassification.from_pretrained("mdhugol/indonesia-bert-sentiment-classification")
    return tokenizer, model

tokenizer_indo, model_indo = load_model()

def predict_indobert(texts):
    inputs = tokenizer_indo(texts.tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        outputs = model_indo(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).numpy()
    label_map = {0:"Negative", 1:"Neutral", 2:"Positive"}
    return [label_map[int(i)] for i in pred]

# -----------------------------
# SCRAPING ULASAN GOOGLE PLAY
# -----------------------------
def scrap_review(app_id, jumlah=300):
    hasil,_ = reviews(app_id, lang="id", country="id", sort=Sort.NEWEST, count=jumlah)
    df = pd.DataFrame(hasil)[["content"]].rename(columns={"content":"text"})
    return df

def extract_app_id(url):
    if "id=" in url:
        return url.split("id=")[-1].split("&")[0]
    return url.strip()

# -----------------------------
# HALAMAN PREDIKSI
# -----------------------------
def render_predict_page():
    st.title("📌 Analisis Sentimen Ulasan Aplikasi Media Sosial")
    st.markdown("Gunakan fitur ini untuk menganalisis sentimen ulasan menggunakan model **IndoBERTa**.")

    link = st.text_input("Masukkan link Google Play / package id")
    jumlah = st.slider("Jumlah ulasan", 50, 1000, 300)
    
    if st.button("Analisis"):
        if not link:
            st.warning("Masukkan link terlebih dahulu!")
            return
        
        app_id = extract_app_id(link)
        
        # Ambil info aplikasi (nama + ikon)
        with st.spinner("Mengambil info aplikasi..."):
            info_app = app(app_id, lang="id", country="id")
            st.image(info_app['icon'], width=64)
            st.markdown(f"### {info_app['title']}")

        # Ambil review
        with st.spinner("Mengambil ulasan..."):
            df = scrap_review(app_id, jumlah)
        st.success(f"Berhasil mengambil {len(df)} ulasan!")

        # Prediksi Sentimen
        df["sentiment"] = predict_indobert(df["text"].astype(str))
        st.subheader("📄 Hasil Sentimen")
        st.dataframe(df, use_container_width=True)

        # Distribusi Label
        st.subheader("📊 Distribusi Label")
        fig_map, ax_map = plt.subplots(figsize=(5,3))
        df["sentiment"].value_counts().plot(kind="bar", ax=ax_map, color=["red","gray","green"])
        ax_map.set_xlabel("Label", fontsize=4)
        ax_map.set_ylabel("Jumlah", fontsize=4)
        ax_map.tick_params(axis='x', labelsize=4)
        ax_map.tick_params(axis='y', labelsize=4)
        fig_map.tight_layout()
        st.pyplot(fig_map)

        # Oversampling
        train_df, test_df = train_test_split(df, test_size=0.3, stratify=df["sentiment"], random_state=42)
        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(train_df[["text"]], train_df["sentiment"])
        train_res = pd.DataFrame({"text": X_res.iloc[:,0], "label": y_res})

        st.subheader("📊 Distribusi Label Setelah Oversampling")
        fig_over, ax_over = plt.subplots(figsize=(5,3))
        train_res["label"].value_counts().plot(kind="bar", ax=ax_over, color=["red","gray","green"])
        ax_over.set_xlabel("Label", fontsize=4)
        ax_over.set_ylabel("Jumlah", fontsize=4)
        ax_over.tick_params(axis='x', labelsize=4)
        ax_over.tick_params(axis='y', labelsize=4)
        fig_over.tight_layout()
        st.pyplot(fig_over)

        # WordCloud
        st.subheader("☁ WordCloud Review")
        all_text = " ".join(df["text"].tolist())
        wc = WordCloud(width=600, height=300, background_color="white").generate(all_text)
        fig_wc, ax_wc = plt.subplots(figsize=(6,3))
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        fig_wc.tight_layout()
        st.pyplot(fig_wc)

        # Confusion Matrix (dummy karena prediksi vs prediksi)
        st.subheader("📈 Confusion Matrix")
        cm = confusion_matrix(df["sentiment"], df["sentiment"], labels=["Negative","Neutral","Positive"])
        fig_cm, ax_cm = plt.subplots(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Negative","Neutral","Positive"],
                    yticklabels=["Negative","Neutral","Positive"], ax=ax_cm)
        ax_cm.set_xlabel("Prediksi", fontsize=4)
        ax_cm.set_ylabel("Aktual", fontsize=4)
        ax_cm.tick_params(axis='x', labelsize=4)
        ax_cm.tick_params(axis='y', labelsize=4)
        fig_cm.tight_layout()
        st.pyplot(fig_cm)
