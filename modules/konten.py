judul = "Analisis Sentimen Ulasan Pengguna Aplikasi Media Sosial di Google Play Store dengan Pendekatan IndoRoBERTa"

string1 = """
Media sosial telah menjadi kebutuhan primer di Indonesia, dengan lebih dari 180 juta pengguna aktif atau sekitar 62,9% dari total populasi. Platform seperti WhatsApp, Instagram, Facebook, dan YouTube mendominasi interaksi digital harian masyarakat. 
Aktivitas yang masif ini menghasilkan jutaan data ulasan di Google Play Store yang merekam pengalaman nyata pengguna, mulai dari kepuasan fitur hingga keluhan teknis. Data ini merupakan aset berharga yang mencerminkan persepsi publik secara real-time.
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
1. Scraping / Input Dara: Pengguna dapat memilih aplikasi yang tersedia (WhatsApp, YouTube, dsb.) atau masukkan Package ID khusus, menentukan jumlah data yang diinginkan.
2. Preprocessing Data: Sistem secara otomatis melakukan preprocessing data berupa lowercasing, data cleaning, dan normalisasi data, serta memberikan label sentimen (Positif, Netral, Negatif).
3. Eksplorasi Data: Pengguna dapat memilih rasio pembagian data. Sistem akan menerapkan Random Oversampling (ROS) agar jumlah data latih seimbang antar kelas sentimen.
4. Pelatihan Model (Fine-Tuning): Pengguna dapat menentukan konfigurasi pelatihan (Epochs, Batch Size, Learning Rate). Klik "Mulai Pelatihan Model" untuk melatih ulang model IndoRoBERTa menggunakan data yang baru diambil.
5. Evaluasi & Uji Coba: Sistem akan memberikan hasil evaluasi dan visualisasi. Setelah selesai, pengguna dapat menginput suatu ulasan untuk menguji coba model.
"""

question1 = "Apa urgensi analisis sentimen pada ulasan pengguna?"
answer1 = """
Analisis sentimen berfungsi untuk **menambang opini** dari teks mentah. 
Menganalisis ribuan ulasan secara manual adalah hal yang mustahil dilakukan secara efisien. Oleh karena itu, sistem analisis sentimen otomatis menjadi sangat krusial karena:
1. Memproses ribuan opini pengguna dalam hitungan detik untuk mengetahui apakah respons publik cenderung positif, negatif, atau netral.
2. Memberikan wawasan tentang kepuasan pengguna untuk meningkatkan pengalaman penggunaan (User Experience) yang lebih baik.
"""

question2 = "Mengapa menggunakan model IndoRoBERTa?"
answer2 = """
**IndoRoBERTa** dipilih karena:
- IndoRoBERTa telah dilatih secara khusus (pre-trained) menggunakan dataset Bahasa Indonesia yang sangat besar.
- Model ini mampu memahami struktur bahasa Indonesia yang kompleks, termasuk penggunaan bahasa gaul (slang), singkatan, dan konteks informal yang sering ditemukan dalam ulasan media sosial.
- Memberikan akurasi klasifikasi yang tinggi untuk teks ulasan yang tidak baku.
"""