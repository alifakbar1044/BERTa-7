judul = "Analisis Sentimen Ulasan Pengguna Aplikasi Media Sosial di Google Play Store dengan Pendekatan IndoBERTa"

string1 = """ 
Ulasan pengguna di Google Play Store merupakan sumber informasi penting karena mencerminkan pengalaman dan kepuasan pengguna terhadap aplikasi media sosial. 
Berdasarkan analisis awal, sebagian besar ulasan bersifat positif, namun terdapat juga ulasan negatif dan netral yang memberikan masukan berharga. 
Analisis sentimen dengan model IndoBERTa memungkinkan pengembang untuk memahami opini pengguna secara otomatis dan efisien.
"""

string2 = """
Sentimen ulasan menjadi indikator penting dalam memahami pengalaman pengguna aplikasi media sosial:
<ul>
<li><b>Sentimen Positif</b> = ulasan yang mengekspresikan kepuasan atau pengalaman menyenangkan dengan aplikasi.</li>
<li><b>Sentimen Negatif</b> = ulasan yang mengekspresikan ketidakpuasan, keluhan, atau masalah yang ditemui pengguna.</li>
<li><b>Sentimen Netral</b> = ulasan yang bersifat informatif atau memberikan saran tanpa emosi kuat.</li>
</ul>
"""

string3 = """
Untuk mengenali pola opini pengguna, sistem ini membuat platform berbasis web menggunakan Streamlit.
Sistem ini memanfaatkan model IndoBERTa untuk melakukan klasifikasi sentimen secara otomatis.
Hasil ditampilkan dalam bentuk grafik distribusi, word cloud, dan tabel ringkasan.
Tujuannya agar pengembang dapat memahami opini pengguna secara visual, efisien, dan mudah dipahami.
"""

cara_penggunaan = """
1. Buka Halaman Analisis Sentimen untuk memulai.
2. Unggah dataset ulasan atau pilih aplikasi dari daftar yang tersedia.
3. Tentukan rentang tanggal ulasan yang ingin dianalisis.
4. Klik 'Jalankan Analisis'.
5. Lihat hasil visualisasi grafik, word cloud, dan tabel ringkasan.
6. Unduh hasil analisis jika diperlukan.
"""

question1 = "Apa itu analisis sentimen pada ulasan pengguna?"
answer1 = """
Analisis sentimen adalah proses untuk mengidentifikasi opini atau perasaan pengguna dalam teks, biasanya dikategorikan menjadi **positif**, **negatif**, atau **netral**. 
Dalam konteks aplikasi media sosial, analisis ini membantu pengembang memahami persepsi pengguna dan mengidentifikasi area untuk perbaikan aplikasi.
"""

question2 = "Metode apa yang digunakan untuk klasifikasi sentimen dalam sistem ini?"
answer2 = """
Sistem ini menggunakan model **IndoBERTa**:
- Mampu memahami konteks bahasa Indonesia dengan baik.
- Mengklasifikasikan ulasan ke dalam sentimen positif, netral, atau negatif.
- Mendukung analisis otomatis untuk dataset besar dengan akurasi tinggi.
"""

penjelasan_dataset = """
- **App_Name**: Nama aplikasi media sosial.
- **Review_ID**: ID ulasan unik.
- **User_Name**: Nama pengguna (anonim/opsional).
- **Review_Text**: Isi ulasan pengguna.
- **Rating**: Rating pengguna (1–5).
- **Date**: Tanggal ulasan ditulis.
- **Sentiment**: Hasil klasifikasi sentimen (Positif/Netral/Negatif).
"""

# PENJELASAN TAMBAHAN
penjelasan_sentimen = "Sentimen Positif, Negatif, dan Netral menunjukkan opini pengguna terhadap aplikasi."
penjelasan_indoberta = "Gunakan IndoBERTa untuk analisis sentimen otomatis dalam bahasa Indonesia."
penjelasan_visualisasi = "Hasil analisis dapat divisualisasikan menggunakan grafik distribusi, word cloud, dan tabel ringkasan."
