import streamlit as st
from streamlit_option_menu import option_menu
import streamlit as st
from halaman.about import render_about_page
from halaman.analysis import render_analysis_page
from halaman.home import render_home_page

def set_mobile_optimization():
    # Untuk layat hp
    st.markdown("""
        <style>
        @media (max-width: 768px) {
            h1 {
                font-size: 24px !important;
                padding-top: 0rem !important;
            }
            h2 {
                font-size: 20px !important;
            }
            h3 {
                font-size: 18px !important;
            }
            p, div, li, span {
                font-size: 14px !important;
            }
            small {
                font-size: 11px !important;
            }
            .block-container {
                padding-top: 2rem !important;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
            [data-testid="stMetricValue"] {
                font-size: 20px !important;
            }
        }
        </style>
    """, unsafe_allow_html=True)

st.set_page_config(
    page_title="IndoRoBERTa Sentiment Analysis",
    page_icon=":robot_face:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- SESSION DEFAULT ---
if "selected_page_index" not in st.session_state:
    st.session_state["selected_page_index"] = 0
    
# --- MENU LIST ---
menu_list = ["Home", "Analysis", "About"]

# --- OPTION MENU ---
selected = option_menu(
    menu_title=None,
    options=menu_list,
    icons=["house", "chat-square-text", "info-circle"],
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#192734"},
        "icon": {"color": "white", "font-size": "12px"},
        "nav-link": {
            "font-size": "12px",
            "color": "white",
            "text-align": "center",
            "margin": "0px",
        },
        "nav-link-selected": {"background-color": "#354687ff", "color": "white"},
    },
    default_index=st.session_state["selected_page_index"],
)

# Jika user klik tab menu, update state dan rerun
current_index = menu_list.index(selected)
if current_index != st.session_state["selected_page_index"]:
    st.session_state["selected_page_index"] = current_index
    st.rerun()

# --- ROUTING ---
page_index = st.session_state["selected_page_index"]

if page_index == 0:
    render_home_page()
elif page_index == 1:
    render_analysis_page()
elif page_index == 2:
    render_about_page()

# --- SIDEBAR GLOSARIUM DAN INFO ---
# 1. Glosarium
with st.sidebar.expander("📖 Glosarium Istilah"):
    st.markdown("""
    <div style="text-align: justify; font-size: 13px;">
    <b>Sentiment Analysis:</b><br>
    Proses komputasi untuk mengidentifikasi dan mengelompokkan opini dalam teks (Positif, Netral, atau Negatif).<br><br>
    <b>IndoRoBERTa:</b><br>
    Model <i>Deep Learning</i> berbasis Transformer yang dilatih khusus untuk memahami konteks Bahasa Indonesia.<br><br>
    <b>Oversampling (ROS):</b><br>
    Teknik duplikasi data pada kelas minoritas agar jumlah data latih menjadi seimbang.<br><br>
    <b>Confusion Matrix:</b><br>
    Tabel evaluasi untuk membandingkan prediksi model dengan label data yang sebenarnya.
    </div>
    """, unsafe_allow_html=True)

st.sidebar.divider()

# 2. Tentang Aplikasi
with st.sidebar.expander("ℹ️ Tentang Sistem"):
    st.markdown("""
    <div style="font-size: 13px;">
    Sistem ini dirancang untuk menganalisis ulasan aplikasi di Google Play Store menggunakan pendekatan <b>IndoRoBERTa</b>.
    <br><br>
    <b>Fitur Utama:</b>
    <ul style="padding-left: 15px; margin-top: 0;">
        <li>Klasifikasi Sentimen Otomatis</li>
        <li>Visualisasi Distribusi Data (Pie/Bar Chart)</li>
        <li>Penyeimbangan Data (Oversampling)</li>
        <li>Analisis Kata Kunci (Word Cloud)</li>
        <li>Evaluasi Performa Model</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)