# BERTa-7
Analisis Sentimen Ulasan Pengguna Aplikasi Media Sosial di Google Play Store dengan Pendekatan IndoBERTa


Panduan Menjalankan Aplikasi BERTa-7 (Streamlit) di Windows
1️⃣ Persiapan Lingkungan

Pastikan Python sudah terinstall.
Cek versi Python:

python --version


Minimal Python 3.10+ direkomendasikan.

Pastikan pip sudah terinstall:

pip --version


Install library yang dibutuhkan (di folder project):

pip install streamlit pandas matplotlib seaborn torch transformers google-play-scraper wordcloud scikit-learn imbalanced-learn plotly


Catatan: Jika menggunakan --user, pastikan Python Scripts sudah ada di PATH, atau jalankan dengan python -m streamlit run app.py.

2️⃣ Masuk ke Folder Project

Misal project ada di C:\Users\alifa\Downloads\BERTa-7:

cd C:\Users\alifa\Downloads\BERTa-7

3️⃣ Menjalankan Aplikasi

Karena kadang streamlit tidak dikenali di PowerShell, gunakan perintah:

python -m streamlit run app.py


Jika berhasil, akan muncul di PowerShell:

Local URL: http://localhost:8501
Network URL: http://192.168.1.18:8501


Buka browser dan akses Local URL untuk melihat aplikasi.