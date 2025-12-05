import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler
import torch
import time

from modules import scraper, preprocessing, modeling

def render_analysis_page():

    st.title("Analisis Sentimen & Eksperimen Model")

    # INPUT DATA
    st.header("1. Konfigurasi Data")
    st.subheader("Pilih Aplikasi (Default)")
    
    use_whatsapp = st.checkbox("WhatsApp (com.whatsapp)", value=True)
    use_youtube = st.checkbox("YouTube (com.google.android.youtube)", value=True)
    use_ig = st.checkbox("Instagram (com.instagram.android)", value=True)
    use_fb = st.checkbox("Facebook (com.facebook.katana)", value=False)

    selected_defaults = []
    if use_whatsapp: selected_defaults.append("com.whatsapp")
    if use_youtube: selected_defaults.append("com.google.android.youtube")
    if use_ig: selected_defaults.append("com.instagram.android")
    if use_fb: selected_defaults.append("com.facebook.katana")

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("Pengaturan Lain")
    html_link = (
        "Contoh link Google Play Store: https:\\"
        "play.google.com/store/apps/details?"
        '<span style="color:#000;">id=</span>'
        '<span style="background-color:#fff59d; padding:2px 6px; border-radius:4px;">com.zhiliaoapp.musically.go</span>'
        '&hl=id'
    )
    st.markdown(html_link, unsafe_allow_html=True)

    custom_app = st.text_input("Package ID Custom (opsional, misal: com.zhiliaoapp.musically.go)", value="")
    data_count = st.slider("Jumlah Data per Aplikasi:", min_value=200, max_value=3000, value=500, step=100)

    if custom_app:
        selected_defaults.append(custom_app)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Mulai Ambil Data (Scraping)", type="primary"):
        if not selected_defaults:
            st.error("Mohon pilih setidaknya satu aplikasi!")
        else:
            start_time = time.time()
            with st.spinner("Sedang mengambil data dari Google Play Store..."):
                df = scraper.scrape_google_play(selected_defaults, count=data_count)
                st.session_state['raw_data'] = df
            end_time = time.time()
            
            duration = end_time - start_time
            st.success(f"Berhasil mengambil {len(df)} ulasan! (Waktu: {duration:.2f} detik)")
            
            
    if 'raw_data' in st.session_state:
        df = st.session_state['raw_data']
        
        with st.expander("Lihat Data Mentah (Hasil Scraping)", expanded=True):
            st.dataframe(df, use_container_width=True, height=300)
            
            col_chart, col_text = st.columns([1, 2])
            with col_chart:
                fig, ax = plt.subplots(figsize=(6, 3))
                try:
                    sns.countplot(x='score', hue='app_id', data=df, palette='viridis', ax=ax)
                    plt.legend(title='Aplikasi', fontsize=8, title_fontsize=9, loc='upper left', bbox_to_anchor=(1, 1))
                    plt.title("Distribusi Rating per Aplikasi", fontsize=10)
                except ValueError:
                    st.warning("⚠️ Kolom 'app_id' tidak ditemukan. Menampilkan grafik total.")
                    sns.countplot(x='score', data=df, palette='viridis', ax=ax)
                    plt.title("Distribusi Rating (Total)", fontsize=10)

                ax.tick_params(axis='both', which='major', labelsize=8)
                plt.tight_layout()
                st.pyplot(fig)
            with col_text:
                st.write(f"**Total Data:** {len(df)} baris")
                st.caption("Grafik di samping menunjukkan sebaran skor bintang (1-5) dari data yang diambil.")

        # PREPROCESSING & LABELING
        st.header("2. Preprocessing & Labeling")
        st.info("""
        **Tahapan Preprocessing yang dilakukan:**
        1. **Mapping Label:** Mengubah Skor Bintang menjadi Label (1-2: Negatif, 3: Netral, 4-5: Positif).
        2. **Case Folding:** Mengubah semua huruf menjadi huruf kecil (lowercase).
        3. **Cleaning:** Menghapus Mention (@user), Hashtag (#), Link (http..), Angka, dan Tanda Baca.
        4. **Normalization:** Menghapus spasi berlebih.
        5. **Stopword Removal:** Menghapus kata hubung umum (yang, di, ke, dll) menggunakan pustaka Sastrawi.
        """)

        if st.button("Jalankan Preprocessing"):
            with st.spinner("Membersihkan teks..."):
                # 1. Mapping Label
                df['label'] = df['score'].apply(preprocessing.map_score_to_label)
                df['label_name'] = df['label'].apply(preprocessing.get_label_name)
                
                # 2. Cleaning
                df = preprocessing.preprocess_dataframe(df)
                
                st.session_state['clean_data'] = df
                st.success("Preprocessing Selesai!")

        if 'clean_data' in st.session_state:
            df_clean = st.session_state['clean_data']
            
            with st.expander("Lihat Data Hasil Preprocessing", expanded=True):
                st.dataframe(df_clean[['content', 'clean_content', 'label', 'label_name']], use_container_width=True, height=300)
                
                col_chart2, col_text2 = st.columns([1, 2])
                with col_chart2:
                    fig2, ax2 = plt.subplots(figsize=(4, 3))
                    sns.countplot(x='label_name', data=df_clean, palette='coolwarm', ax=ax2)
                    plt.title("Distribusi Sentimen Awal", fontsize=10)
                    ax2.tick_params(labelsize=8)
                    st.pyplot(fig2)

            # PEMBAGIAN DATA & OVERSAMPLING
            st.header("3. Pembagian Data & Oversampling")
            
            st.subheader("Konfigurasi Rasio Data")
            split_option = st.selectbox(
                "Pilih Rasio Pembagian (Train : Validation : Test)",
                options=["80 : 10 : 10", "70 : 15 : 15", "60 : 20 : 20"],
                index=0
            )
            
            if split_option == "80 : 10 : 10":
                test_ratio = 0.10
                val_ratio = 0.10
            elif split_option == "70 : 15 : 15":
                test_ratio = 0.15
                val_ratio = 0.15
            else: # 60 : 20 : 20
                test_ratio = 0.20
                val_ratio = 0.20

            # Split Data
            X = df_clean['clean_content'].values
            y = df_clean['label'].values
            
            # Tahap 1: Pisahkan Data Test dari Total
            # (Misal 80:10:10 -> Ambil 10% untuk Test, sisa 90% masuk Train_Full)
            X_train_full, X_test, y_train_full, y_test = train_test_split(
                X, y, test_size=test_ratio, stratify=y, random_state=42
            )
            
            # Tahap 2: Pisahkan Data Validation dari Train_Full
            val_ratio_relative = val_ratio / (1 - test_ratio)

            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full, test_size=val_ratio_relative, stratify=y_train_full, random_state=42
            )

            # Oversampling
            ros = RandomOverSampler(random_state=42)
            X_train_ros, y_train_ros = ros.fit_resample(X_train.reshape(-1, 1), y_train)
            X_train_ros = X_train_ros.flatten()
            
            total_data = len(X)
            
            # Menampilkan Metrik
            c_split1, c_split2, c_split3 = st.columns(3)
            
            train_pct = len(X_train)/total_data
            val_pct = len(X_val)/total_data
            test_pct = len(X_test)/total_data

            c_split1.metric(
                "Data Latih (Train)", 
                f"{len(X_train)} ({train_pct:.1%})", 
                "Akan di-oversampling"
            )
            c_split2.metric(
                "Data Validasi", 
                f"{len(X_val)} ({val_pct:.1%})"
            )
            c_split3.metric(
                "Data Uji (Test)", 
                f"{len(X_test)} ({test_pct:.1%})"
            )

            st.info(f"**Status Oversampling:** Jumlah data latih meningkat dari **{len(X_train)}** menjadi **{len(X_train_ros)}** agar seimbang.")

            col_chart3, col_text3 = st.columns([1, 2])
            with col_chart3:
                fig3, ax3 = plt.subplots(figsize=(4, 3))
                sns.countplot(x=y_train_ros, palette='coolwarm', ax=ax3)
                ax3.set_xticklabels(['Negatif', 'Netral', 'Positif'])
                plt.title("Distribusi Label Latih (Balanced)", fontsize=10)
                ax3.tick_params(labelsize=8)
                st.pyplot(fig3)
            
            # MODELING (INDOROBERTA)
            st.header("4. Fine-Tuning IndoRoBERTa")
            st.markdown("**Base Model:** `flax-community/indonesian-roberta-base` (Hugging Face)")
            
            with st.form("hyperparam_form"):
                st.subheader("Atur Hyperparameter")
                
                c1, c2, c3 = st.columns(3)
                epochs = c1.number_input("Epochs", min_value=1, max_value=10, value=2)
                c1.caption("Default: 2. Jumlah putaran pelatihan.")
                
                batch_size = c2.selectbox("Batch Size", [8, 16, 32, 64], index=0)
                c2.caption("Default: 8. Semakin besar butuh memori GPU besar.")
                
                lr = c3.selectbox("Learning Rate", [1e-5, 2e-5, 3e-5, 5e-5], index=1)
                c3.caption("Default: 2e-5. Kecepatan model belajar.")
                
                run_train = st.form_submit_button("Mulai Pelatihan Model", type="primary")
                
            if run_train:
                hyperparams = {'epochs': epochs, 'batch_size': batch_size, 'lr': lr}
                
                start_train = time.time()
                with st.spinner("Sedang melatih IndoRoBERTa... (Proses ini memakan waktu)"):
                    try:
                        # PANGGIL FUNGSI TRAINING
                        trainer, tokenizer, model = modeling.train_indoroberta(
                            X_train_ros, y_train_ros, X_val, y_val, hyperparams
                        )
                        
                        end_train = time.time()
                        duration_train = end_train - start_train
                        
                        st.success(f"Pelatihan Selesai! Waktu: {duration_train:.2f} detik")
                        
                        st.session_state['trainer'] = trainer
                        st.session_state['tokenizer'] = tokenizer
                        st.session_state['model'] = model
                        st.session_state['test_data'] = (X_test, y_test)
                        
                    except Exception as e:
                        st.error(f"Terjadi error saat training: {e}")
                        print(f"DEBUG ERROR: {e}")

            # EVALUASI & VISUALISASI AKHIR
            if 'trainer' in st.session_state:
                st.header("5. Evaluasi & Hasil Akhir")
                
                trainer = st.session_state['trainer']
                tokenizer = st.session_state['tokenizer']
                X_test, y_test = st.session_state['test_data']
                
                with st.spinner("Melakukan prediksi pada data uji..."):
                    test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=128)
                    test_dataset = modeling.SentimentDataset(test_encodings, list(y_test))
                    
                    predictions = trainer.predict(test_dataset)
                    preds = np.argmax(predictions.predictions, axis=-1)
                
                # Metrics
                acc = (preds == y_test).mean()
                report = classification_report(y_test, preds, target_names=['Negatif', 'Netral', 'Positif'], output_dict=True)
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Akurasi", f"{acc:.2%}")
                m2.metric("Precision", f"{report['macro avg']['precision']:.2f}")
                m3.metric("Recall", f"{report['macro avg']['recall']:.2f}")
                m4.metric("F1-Score", f"{report['macro avg']['f1-score']:.2f}")
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                col_cm, _ = st.columns([1, 1])
                with col_cm:
                    cm = confusion_matrix(y_test, preds)
                    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                xticklabels=['Neg', 'Net', 'Pos'], 
                                yticklabels=['Neg', 'Net', 'Pos'], ax=ax_cm)
                    plt.ylabel('Aktual')
                    plt.xlabel('Prediksi')
                    st.pyplot(fig_cm)
                
                # WordCloud
                st.subheader("WordCloud Hasil Prediksi")
                wc_col1, wc_col2, wc_col3 = st.columns(3)
                df_res = pd.DataFrame({'text': X_test, 'pred': preds, 'actual': y_test})
                
                def show_wc(label_code, title, col):
                    text_wc = " ".join(df_res[df_res['pred'] == label_code]['text'].values)
                    if text_wc:
                        wc = WordCloud(width=300, height=200, background_color='white').generate(text_wc)
                        col.image(wc.to_array(), caption=title)
                    else:
                        col.write(f"Tidak ada data untuk {title}")

                show_wc(0, "Prediksi Negatif", wc_col1)
                show_wc(1, "Prediksi Netral", wc_col2)
                show_wc(2, "Prediksi Positif", wc_col3)

                # Tabel Hasil Akhir
                st.subheader("Tabel Detail Hasil Prediksi")
                df_res['label_pred'] = df_res['pred'].apply(preprocessing.get_label_name)
                df_res['label_actual'] = df_res['actual'].apply(preprocessing.get_label_name)
                
                st.dataframe(df_res, use_container_width=True, height=400)
                
                csv = df_res.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Hasil Prediksi (CSV)",
                    data=csv,
                    file_name='hasil_prediksi_sentimen.csv',
                    mime='text/csv',
                )