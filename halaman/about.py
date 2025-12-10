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
    
    1.  **Input Data**: Mengambil data ulasan terbaru secara langsung (real-time) dari Google Play Store berdasarkan ID aplikasi yang dipilih (seperti WhatsApp, Instagram) atau ID kustom.
    2.  **Preprocessing Teks & Labeling Data otomatis**: Mengonversi rating bintang pengguna menjadi label sentimen (Skor 1-2: Negatif, 3: Netral, 4-5: Positif). Lalu, Membersihkan teks dari karakter non-huruf, mentions, hashtags, serta normalisasi bahasa gaul (slang) menjadi bahasa baku menggunakan kamus leksikon.
    3.  **Pembagian & Penyeimbangan Data**: Menerapkan data splitting menjadi data training, validasi, dan data uji. Juga menerapkan teknik **Oversampling (ROS)** untuk mengatasi ketidakseimbangan jumlah data antar kelas sentimen.
    4.  **Modeling (IndoRoBERTa)**: Melatih model pre-trained IndoRoBERTa menggunakan data ulasan yang telah diproses. Pengguna dapat mengatur hyperparameters (Epochs, Batch Size, Learning Rate) untuk mengoptimalkan performa model.
    5.  **Evaluasi & Visualisasi**: Menampilkan hasil analisis dalam bentuk:
        * *Metrik Evaluasi* Akurasi, Presisi, Recall, dan F1-Score.
        * *Word Cloud* untuk melihat kata yang paling sering muncul.
    6. Fitur tambahan: Uji coba prediksi, pengguna dapat memasukkan kalimat ulasan untuk menguji hasil prediksi.
    """)

    st.divider()

    # BAGIAN 3: MENGAPA INDOROBERTA
    st.subheader("🧠 Mengapa IndoRoBERTa?")
    st.markdown("""
    Sistem ini memanfaatkan **IndoRoBERTa**, model berbasis *Transformer* yang unggul dalam memahami konteks semantik Bahasa Indonesia dibandingkan metode tradisional (seperti Naive Bayes atau SVM). 
    Model ini merupakan varian dari IndoBERT
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