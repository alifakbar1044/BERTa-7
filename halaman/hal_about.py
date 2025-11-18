import streamlit as st

def render_about_page():
    st.title("ℹ Tentang BERTa-7")
    
    st.markdown("""
    **BERTa-7** adalah aplikasi analisis sentimen untuk ulasan aplikasi Google Play.

    **Fitur utama BERTa-7**:
    - Prediksi Sentimen: Positif, Netral, Negatif
    - Distribusi Label & Oversampling
    - WordCloud Review
    - Confusion Matrix
    - Analisis otomatis menggunakan model **IndoBERTa** (Pre-trained BERT Bahasa Indonesia)
    """)
    
    st.markdown("---")
    st.subheader("Tentang IndoBERTa")
    st.markdown("""
    **IndoBERTa** adalah model Bahasa Indonesia berbasis BERT yang digunakan untuk:
    - Klasifikasi sentimen ulasan pengguna
    - Pemrosesan teks Bahasa Indonesia
    - Analisis opini secara otomatis dan efisien
    """)
    
    st.info("BERTa-7 mempermudah pengembang dan peneliti memahami opini pengguna secara cepat dan akurat melalui analisis sentimen.")
