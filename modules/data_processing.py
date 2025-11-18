# modules/data_processing.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import streamlit as st

# ===============================
# 1️⃣ Muat Data
# ===============================
def muat_data(path, sheet=None):
    """
    Muat data dari CSV atau Excel.
    Parameter sheet hanya untuk Excel.
    """
    try:
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path, sheet_name=sheet)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{path}' tidak ditemukan.")
    except Exception as e:
        raise Exception(f"Gagal memuat data: {e}")

# ===============================
# 2️⃣ Preprocessing
# ===============================
def replace_non_numeric(df):
    """
    Mengganti nilai non-numeric menjadi NaN pada kolom numeric
    """
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            try:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='ignore')
            except:
                pass
    return df_copy

def del_col_non_numeric(df, keep_columns=[]):
    """
    Menghapus kolom non-numeric kecuali kolom penting
    """
    df_copy = df.copy()
    non_numeric_cols = df_copy.select_dtypes(include='object').columns
    for col in non_numeric_cols:
        if col not in keep_columns:
            df_copy.drop(columns=col, inplace=True)
    return df_copy

def fill_missing(df, method='ffill'):
    """
    Mengisi missing value
    method: 'ffill', 'bfill', atau 'mean'
    """
    df_copy = df.copy()
    if method == 'ffill':
        df_copy = df_copy.ffill()
    elif method == 'bfill':
        df_copy = df_copy.bfill()
    elif method == 'mean':
        df_copy = df_copy.fillna(df_copy.mean())
    return df_copy

# ===============================
# 3️⃣ Normalisasi
# ===============================
def data_normalization(df):
    """
    Normalisasi kolom numeric
    """
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        mean = df_copy[col].mean()
        std = df_copy[col].std()
        if std != 0:
            df_copy[col] = (df_copy[col] - mean) / std
        else:
            df_copy[col] = 0
    return df_copy

# ===============================
# 4️⃣ Clustering
# ===============================
def kmeans_clustering(df, n_clusters):
    """
    Menjalankan K-Means
    """
    if df.empty or n_clusters is None:
        st.warning("Data kosong atau jumlah cluster tidak ditentukan")
        return None

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(df)
    return {"labels": labels, "centroids": kmeans.cluster_centers_}

def dbscan_clustering(df, eps=0.5, min_samples=5):
    """
    Menjalankan DBSCAN
    """
    if df.empty:
        st.warning("Data kosong untuk DBSCAN")
        return None

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='manhattan')
    labels = dbscan.fit_predict(df)
    core_indices = dbscan.core_sample_indices_

    point_type = np.full(len(labels), "Border", dtype=object)
    point_type[core_indices] = "Core"
    point_type[labels == -1] = "Noise"

    return {"labels": labels, "point_type": point_type.tolist()}

# ===============================
# 5️⃣ Preprocessing Lengkap
# ===============================
def preprocessing_sentiment_data(df, keep_columns=['App_Name', 'Review_Text']):
    """
    Pipeline preprocessing untuk dataset ulasan Google Play
    """
    df_numeric = del_col_non_numeric(df, keep_columns=keep_columns)
    df_clean = replace_non_numeric(df_numeric)
    df_filled = fill_missing(df_clean, method='mean')
    df_norm = data_normalization(df_filled)
    return df_numeric, df_clean, df_filled, df_norm
