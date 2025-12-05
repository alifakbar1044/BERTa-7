import streamlit as st

def render_about_page():
    st.title("ℹ️ Tentang RoBERTa-A7")
    
    # BAGIAN 1: DESKRIPSI PROJEK UAS
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 25px; border-left: 5px solid #ff4b4b;">
        <h5>🎓 Proyek Ujian Akhir Semester (UAS)</h5>
        <strong>Mata Kuliah:</strong> Natural Language Processing (NLP)<br>
        <strong>Topik:</strong> Analisis Sentimen Ulasan Aplikasi Menggunakan Deep Learning
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # BAGIAN 2: METODOLOGI SISTEM
    st.subheader("🛠️ Bagaimana Sistem Bekerja?")
    st.markdown("""
    **RoBERTa-A7** adalah sistem *end-to-end* yang dirancang untuk mengekstrak wawasan dari ulasan Google Play Store. Berikut adalah tahapan proses yang dilakukan oleh sistem ini:
    
    1.  **Akuisisi Data**: Menerima input dataset ulasan (format `.csv` atau `.xlsx`) yang berisi teks komentar pengguna.
    2.  **Preprocessing Teks**: Membersihkan data mentah (menghapus emoji, tanda baca, *case folding*) agar siap diproses oleh mesin.
    3.  **Penyeimbangan Data**: Menerapkan teknik **Oversampling (ROS)** untuk mengatasi ketidakseimbangan jumlah data antar kelas sentimen.
    4.  **Modeling (IndoRoBERTa)**: Menggunakan *Pre-trained Model* **IndoRoBERTa** yang telah dilatih pada korpus Bahasa Indonesia masif untuk mengklasifikasikan sentimen (Positif, Netral, Negatif).
    5.  **Visualisasi Interaktif**: Menampilkan hasil analisis dalam bentuk:
        * *Pie/Bar Chart* untuk distribusi sentimen.
        * *Word Cloud* untuk melihat kata yang paling sering muncul.
        * *Confusion Matrix* untuk evaluasi akurasi model.
    """)

    st.divider()

    # BAGIAN 3: MENGAPA INDOROBERTA
    st.subheader("🧠 Mengapa IndoRoBERTa?")
    st.markdown("""
    Sistem ini memanfaatkan **IndoRoBERTa**, model berbasis *Transformer* yang unggul dalam memahami konteks semantik Bahasa Indonesia dibandingkan metode tradisional (seperti Naive Bayes atau SVM). 
    
    Model ini mampu menangkap nuansa bahasa gaul (*slang*), singkatan, dan struktur kalimat kompleks yang sering ditemukan pada ulasan media sosial.
    """)

    st.divider()

    # BAGIAN 4: TIM PENGEMBANG
    st.subheader("👥 Tim Pengembang")
    st.markdown("Sistem ini dikembangkan oleh kelompok mahasiswa:")
    
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### 👩‍💻 Anggota 1")
        st.markdown("**Jessica Ho**")
        st.caption("NIM: 535220187")

    with c2:
        st.markdown("### 👩‍💻 Anggota 2")
        st.markdown("**Parveen Uzma**")
        st.caption("NIM: 535220226")
    with c3:
        st.markdown("### 👨‍💻 Anggota 3")
        st.markdown("**Muhammad Akbar**")
        st.caption("NIM: 535230023")
    
    st.divider()
    
    # BAGIAN 5: TECH STACK
    st.subheader("⚙️ Tech Stack")
    st.markdown("""
        Sistem ini dibangun dengan:

        * **Bahasa Pemrograman:** Python 3.10+
        * **Framework:** :streamlit:[Streamlit](https://streamlit.io)
        * **Model IndoRoBERTa:** [IndoRoBERTa](https://huggingface.co/indolem/indoberta-base-uncased)
        * **Library:**
            * Pandas & Numpy untuk manipulasi data
            * Scikit-Learn (Evaluasi & Splitting) & Imbalanced-learn (Oversampling)
            * Matplotlib, Seaborn, & WordCloud untuk visualisasi
            * Transformers & Accelerate untuk model
            * Google Play Scraper untuk ekstraksi data
    """)
    
    st.write("")
    
    
    st.success("© 2025 RoBERTa-A7 Project. All Rights Reserved.")