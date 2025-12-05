# RoBERTa-A7
Analisis Sentimen Ulasan Pengguna Aplikasi Media Sosial di Google Play Store dengan Pendekatan IndoBERT

## 📖 Deskripsi Proyek
**RoBERTa-A7** adalah sistem analisis sentimen berbasis web untuk menganalisis sentimen dari ulasan aplikasi di Google Play Store. Sistem ini menggunakan **IndoRoBERTa** (Transformer) untuk mengklasifikasikan opini pengguna ke dalam sentimen **Positif**, **Netral**, atau **Negatif** secara otomatis. Proyek ini dibuat untuk memenuhi tugas Ujian Akhir Semester (UAS) mata kuliah Natural Language Processing.

Anggota:
* 535220187 - Jessica Ho
* 535220226 - Parveen Uzma
* 535230023 - Muhammad Akbar
---

Tech Stack
* Python 3.10+
* Streamlit (Web Framework)
* Transformers
* Scikit-Learn & Imbalanced-learn
* Matplotlib & Seaborn
* Selengkapnya dapat dilihat pada [Requirements](requirements.txt)
---

## Panduan Menjalankan Aplikasi RoBERTa-A7 (Streamlit) di Windows

### 1. Persiapan Lingkungan
- Pastikan Python sudah terinstall (disarankan Python 3.10+)

  Cek versi Python:
  ```bash
  python --version
  ```
  atau
  ```bash
  py --version
  ```
  
  Pastikan pip sudah terinstall:
  ```bash
  pip --version
  ```

### 2. Masuk ke Folder Project
Misal project ada di `C:\Users\alifa\Downloads\BERTa-7`:

```bash
cd C:\Users\alifa\Downloads\BERTa-7
```

### 3. Membuat Virtual Environment
- Buat venv di folder project (pilih salah satu sesuai Python di PC):
  ```bash
  python -m venv venv
  ```
  atau
  ```bash
  py -m venv venv
  ```

- Aktifkan venv:
  ```bash
  .\venv\Scripts\Activate.ps1
  ```

- Install library yang dibutuhkan:
  ```bash
  pip install -r requirements.txt
  ```

### 4. Menjalankan Aplikasi (Pilih salah satu sesuai Python di PC)
```bash
python -m streamlit run app.py
```
atau
```bash
py -m streamlit run app.py
```

Jika berhasil, akan muncul di PowerShell:

```
Local URL: http://localhost:8501
Network URL: http://192.168.1.18:8501
```

Buka browser dan akses **Local URL** untuk melihat aplikasi.
