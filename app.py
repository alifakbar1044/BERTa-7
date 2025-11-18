# app.py
import streamlit as st
from streamlit_option_menu import option_menu
from halaman.hal_home import render_home_page
from halaman.hal_predict import render_predict_page
from halaman.hal_dataset import render_dataset_page
from halaman.hal_about import render_about_page

# --- CONFIG PAGE ---
st.set_page_config(
    page_title="BERTa-7 – Prediksi Sentimen",
    page_icon=":speech_balloon:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- SESSION DEFAULT ---
if "selected_page_index" not in st.session_state:
    st.session_state["selected_page_index"] = 0

# --- MENU LIST ---
menu_list = ["Home", "Dataset", "Analisis", "Tentang"]

# --- OPTION MENU HORIZONTAL DENGAN FONT KECIL ---
selected = option_menu(
    menu_title=None,
    options=menu_list,
    icons=["house", "database", "chat-square-text", "info-circle"],
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#192734"},
        "icon": {"color": "white", "font-size": "12px"},  # icon kecil
        "nav-link": {
            "font-size": "12px",  # font menu kecil
            "color": "white",
            "text-align": "center",
            "margin": "0px",
        },
        "nav-link-selected": {"background-color": "#215C3B", "color": "white"},
    },
    default_index=st.session_state["selected_page_index"],
)

# --- UPDATE SESSION STATE ---
current_index = menu_list.index(selected)
if current_index != st.session_state["selected_page_index"]:
    st.session_state["selected_page_index"] = current_index
    st.rerun()

# --- ROUTING HALAMAN ---
page_index = st.session_state["selected_page_index"]
if page_index == 0:
    render_home_page()
elif page_index == 1:
    render_dataset_page()
elif page_index == 2:
    render_predict_page()
elif page_index == 3:
    render_about_page()

# --- SIDEBAR GLOSARIUM DAN INFO INDO BERTA ---
st.sidebar.divider()
with st.sidebar.expander("📖 Glosarium Istilah"):
    st.markdown("""
    <small>
    **Sentiment Analysis**: Analisis teks untuk mengidentifikasi sentimen (Positif, Netral, Negatif).<br>
    **BERTa-7**: Model Bahasa Indonesia berbasis BERT untuk klasifikasi sentimen.<br>
    **WordCloud**: Visualisasi kata yang sering muncul pada ulasan.<br>
    **Confusion Matrix**: Matriks evaluasi performa model klasifikasi.<br>
    **Oversampling**: Teknik menyeimbangkan jumlah data antar kelas.
    </small>
    """, unsafe_allow_html=True)

st.sidebar.divider()
with st.sidebar.expander("ℹ Tentang IndoBERTa & Fitur BERTa-7"):
    st.markdown("""
    <small>
    **IndoBERTa**: Pre-trained model Bahasa Indonesia berbasis BERT, digunakan untuk analisis sentimen ulasan aplikasi.<br>
    **Fitur BERTa-7**:<br>
    - Prediksi Sentimen: Positif, Netral, Negatif<br>
    - Distribusi Label & Oversampling<br>
    - WordCloud Review<br>
    - Confusion Matrix<br>
    - Analisis Sentimen otomatis berbasis model IndoBERTa
    </small>
    """, unsafe_allow_html=True)
