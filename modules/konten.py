judul = "Analisis Sentimen Ulasan Pengguna Aplikasi Media Sosial di Google Play Store dengan Pendekatan IndoRoBERTa"

string1 = """
Ulasan pengguna di Google Play Store merupakan aset vital yang mencerminkan tingkat kepuasan dan pengalaman nyata terhadap aplikasi media sosial. 
Meskipun mayoritas ulasan cenderung positif, ulasan negatif dan netral sering kali menyimpan wawasan krusial terkait bug, keluhan, atau saran perbaikan. 
Sistem ini menggunakan model **IndoRoBERTa** untuk mengotomatisasi proses analisis tersebut, sehingga pola opini pengguna dapat dipahami secara lebih efisien dan akurat.
"""

string2 = """
Sentimen ulasan dikategorikan menjadi tiga indikator utama untuk memetakan persepsi pengguna:
<ul>
<li><b>Sentimen Positif:</b> Merepresentasikan kepuasan, pujian, atau pengalaman penggunaan yang menyenangkan.</li>
<li><b>Sentimen Negatif:</b> Mencerminkan ketidakpuasan, keluhan teknis, atau kendala yang dialami pengguna.</li>
<li><b>Sentimen Netral:</b> Berisi informasi objektif, pertanyaan, atau saran fitur tanpa muatan emosi yang kuat.</li>
</ul>
"""

string3 = """
Platform ini dikembangkan menggunakan kerangka kerja **Streamlit** untuk memvisualisasikan hasil analisis secara interaktif.
Dengan mengintegrasikan model *Deep Learning* IndoRoBERTa, sistem mampu mengklasifikasikan sentimen secara otomatis.
Hasil analisis disajikan melalui **Grafik Distribusi**, **Word Cloud**, dan **Tabel Ringkasan** untuk memudahkan pengembang dalam mengambil keputusan berbasis data.
"""

cara_penggunaan = """
1. **Mulai:** Buka menu 'Analisis Sentimen' pada sidebar.
2. **Input Data:** Unggah dataset ulasan baru atau pilih sampel aplikasi yang tersedia.
3. **Filter:** Tentukan rentang tanggal ulasan yang ingin dianalisis (opsional).
4. **Proses:** Klik tombol 'Jalankan Analisis' dan tunggu model bekerja.
5. **Eksplorasi:** Analisis visualisasi melalui grafik, word cloud, dan metrik yang muncul.
6. **Simpan:** Unduh hasil klasifikasi dalam format CSV/Excel jika diperlukan.
"""

question1 = "Apa urgensi analisis sentimen pada ulasan pengguna?"
answer1 = """
Analisis sentimen berfungsi untuk **menambang opini** dari teks mentah. 
Dalam konteks pengembangan aplikasi, ini membantu tim developer untuk mendeteksi bug kritis lebih cepat, memahami fitur yang paling disukai, dan merespons keluhan pengguna berdasarkan data, bukan asumsi.
"""

question2 = "Mengapa menggunakan model IndoRoBERTa?"
answer2 = """
**IndoRoBERTa** dipilih karena:
- Merupakan model berbasis Transformer yang telah dilatih khusus pada korpus Bahasa Indonesia yang masif.
- Memiliki kemampuan memahami **konteks kalimat**, slang, dan singkatan yang sering muncul di media sosial jauh lebih baik daripada metode tradisional.
- Memberikan akurasi klasifikasi yang tinggi untuk teks ulasan yang tidak baku.
"""

penjelasan_dataset = """
Pastikan dataset yang diunggah memiliki format kolom berikut:
- **App_Name**: Nama aplikasi (String).
- **Review_Text**: Isi ulasan pengguna yang akan dianalisis (String).
- **Rating**: Skala penilaian 1-5 (Integer).
- **Date**: Tanggal ulasan (Datetime).
- *(Opsional)* **User_Name**: Nama pengguna.
"""

# PENJELASAN TAMBAHAN
penjelasan_sentimen = "Distribusi ini menunjukkan persentase opini Positif, Negatif, dan Netral dalam dataset."
penjelasan_indoRoBERTa = "Model IndoRoBERTa memproses teks untuk menangkap nuansa bahasa Indonesia secara mendalam."
penjelasan_visualisasi = "Visualisasi interaktif untuk memudahkan pemahaman pola data secara cepat."