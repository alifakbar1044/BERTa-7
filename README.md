# IndoBERT(A-7)
Analisis Sentimen Ulasan Pengguna Aplikasi Media Sosial di Google Play Store dengan Pendekatan IndoBERT

Anggota:
* 535220187 - Jessica Ho
* 535220226 - Parveen Uzma
* 535230023 - Muhammad Akbar
---

## Panduan Menjalankan Aplikasi IndoBERT(A-7) (Streamlit) di Windows

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

- Aktifkan venv (Windows):
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
