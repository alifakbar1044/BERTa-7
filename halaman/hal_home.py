import streamlit as st
from modules.konten import judul, string1, string2, string3, cara_penggunaan, question1, question2, answer1, answer2

def render_home_page():    
    st.markdown("""
    <style>
    h1 {
        text-align: center;
    }

    @media (max-width: 600px) {
        h1 {
            font-size: 1.6rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    st.title(judul)
    
    st.divider()
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("📊 Lihat Dataset", use_container_width=True):
            st.session_state['selected_page_index'] = 1
            st.rerun()
    with col_btn2:
        if st.button("🚀 Mulai Analisis", type="primary", use_container_width=True):
            st.session_state['selected_page_index'] = 2
            st.rerun()
    
    st.write("")
    
    # Cara penggunaan singkat
    st.subheader("CARA PENGGUNAAN SINGKAT")
    st.markdown(cara_penggunaan, unsafe_allow_html=True)
    st.write("")
    
    # Latar Belakang
    st.subheader("LATAR BELAKANG")
    st.markdown(string1, unsafe_allow_html=True)
    st.write("")
    
    # Deskripsi Sistem
    st.markdown(string2, unsafe_allow_html=True)
    st.write("")
    st.markdown(string3, unsafe_allow_html=True)
    st.write("")
    
    # FAQ
    st.subheader("FAQ")
    st.expander(question1, expanded=False).markdown(answer1, unsafe_allow_html=True)
    st.expander(question2, expanded=False).markdown(answer2, unsafe_allow_html=True)
