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
menu_list = ["Home", "Dataset", "Prediksi", "Tentang"]

# --- OPTION MENU HORIZONTAL ---
selected = option_menu(
    menu_title=None,
    options=menu_list,
    icons=["house", "database", "chat-square-text", "info-circle"],
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#192734"},
        "icon": {"color": "white", "font-size": "18px"},
        "nav-link": {
            "font-size": "18px",
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
    **Sentiment Analysis**: Analisis teks untuk mengidentifikasi sentimen (Positif, Netral, Negatif).

    **BERTa-7**: Model Bahasa Indonesia berbasis BERT untuk klasifikasi sentimen.

    **WordCloud**: Visualisasi kata yang sering muncul pada ulasan.

    **Confusion Matrix**: Matriks evaluasi performa model klasifikasi.

    **Oversampling**: Teknik menyeimbangkan jumlah data antar kelas.
    """)

st.sidebar.divider()
with st.sidebar.expander("ℹ Tentang IndoBERTa & Fitur BERTa-7"):
    st.markdown("""
    **IndoBERTa**: Pre-trained model Bahasa Indonesia berbasis BERT, digunakan untuk analisis sentimen ulasan aplikasi.  

    **Fitur BERTa-7**:
    - Prediksi Sentimen: Positif, Netral, Negatif
    - Distribusi Label & Oversampling
    - WordCloud Review
    - Confusion Matrix
    - Analisis Sentimen otomatis berbasis model IndoBERTa
    """)
