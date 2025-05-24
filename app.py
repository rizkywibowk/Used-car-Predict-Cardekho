# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import pickle

# --- FUNGSI UNTUK MEMUAT MODEL DAN DATA PENDUKUNG ---
@st.cache_resource # Cache resource agar tidak load ulang setiap interaksi
def load_resources():
    try:
        model = joblib.load('model.joblib')
    except FileNotFoundError:
        st.error("File model 'model.joblib' tidak ditemukan. Jalankan 'train_model.py' terlebih dahulu.")
        return None, None, None
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None, None, None

    try:
        with open('training_columns.pkl', 'rb') as f:
            training_columns = pickle.load(f)
    except FileNotFoundError:
        st.error("File 'training_columns.pkl' tidak ditemukan. Jalankan 'train_model.py' terlebih dahulu.")
        return model, None, None # Kembalikan model jika ada, tapi kolom tidak
    except Exception as e:
        st.error(f"Error saat memuat kolom training: {e}")
        return model, None, None

    try:
        with open('input_options.pkl', 'rb') as f:
            input_options = pickle.load(f)
    except FileNotFoundError:
        st.error("File 'input_options.pkl' tidak ditemukan. Jalankan 'train_model.py' terlebih dahulu.")
        return model, training_columns, None # Kembalikan model & kolom jika ada
    except Exception as e:
        st.error(f"Error saat memuat opsi input: {e}")
        return model, training_columns, None

    return model, training_columns, input_options

model, training_columns, input_options = load_resources()

# --- ANTARMUKA PENGGUNA STREAMLIT ---
st.set_page_config(page_title="Prediksi Harga Mobil Bekas", layout="wide")
st.title("🚗 Prediksi Harga Mobil Bekas")
st.markdown("Masukkan detail mobil untuk mendapatkan estimasi harga jual.")

if model is None or training_columns is None or input_options is None:
    st.warning("Aplikasi tidak dapat berjalan karena resource penting (model/kolom/opsi input) gagal dimuat. Silakan periksa pesan error di atas.")
else:
    # --- INPUT DARI PENGGUNA DI SIDEBAR ---
    st.sidebar.header("Masukkan Fitur Mobil:")

    # Mendapatkan nilai default atau list kosong jika key tidak ada di input_options
    available_car_names = input_options.get('name', ['Tidak Ada Pilihan'])
    available_fuel_types = input_options.get('fuel', ['Tidak Ada Pilihan'])
    available_seller_types = input_options.get('seller_type', ['Tidak Ada Pilihan'])
    available_transmissions = input_options.get('transmission', ['Tidak Ada Pilihan'])
    available_owners = input_options.get('owner', ['Tidak Ada Pilihan'])

    # Ambil tahun saat ini untuk batas input tahun
    current_system_year = datetime.now().year

    car_name = st.sidebar.selectbox("Model Mobil (Name)", options=available_car_names)
    year = st.sidebar.number_input("Tahun Pembuatan", min_value=1980, max_value=current_system_year, value=current_system_year - 5, step=1)
    km_driven = st.sidebar.number_input("Jarak Tempuh (KM)", min_value=0, value=50000, step=1000)
    fuel = st.sidebar.selectbox("Jenis Bahan Bakar", options=available_fuel_types)
    seller_type = st.sidebar.selectbox("Jenis Penjual", options=available_seller_types)
    transmission = st.sidebar.selectbox("Jenis Transmisi", options=available_transmissions)
    owner = st.sidebar.selectbox("Kepemilikan", options=available_owners)
    
    # Fitur numerik tambahan
    mileage_input = st.sidebar.number_input("Konsumsi BBM (km/l atau km/kg)", min_value=0.0, value=18.0, step=0.1, format="%.1f")
    engine_input = st.sidebar.number_input("Kapasitas Mesin (CC)", min_value=0, value=1200, step=50)
    max_power_input = st.sidebar.number_input("Tenaga Maksimal (bhp)", min_value=0.0, value=80.0, step=1.0, format="%.1f")
    seats_input = st.sidebar.number_input("Jumlah Kursi", min_value=1, max_value=10, value=5, step=1)

    # Tombol Prediksi
    if st.sidebar.button("🔮 Prediksi Harga"):
        # --- MEMBUAT DATAFRAME DARI INPUT PENGGUNA ---
        input_data = {
            'name': [car_name], # Perlu di one-hot encode nanti
            'km_driven': [km_driven],
            'fuel': [fuel], # Perlu di one-hot encode
            'seller_type': [seller_type], # Perlu di one-hot encode
            'transmission': [transmission], # Perlu di one-hot encode
            'owner': [owner], # Perlu di one-hot encode
            'mileage': [mileage_input], # Sudah di-rename di training
            'engine': [engine_input],
            'max_power': [max_power_input],
            'seats': [seats_input],
            'car_age': [current_system_year - year] # Feature engineering untuk input
        }
        input_df = pd.DataFrame.from_dict(input_data)

        # --- PRA-PEMROSESAN INPUT PENGGUNA (MIRIP DENGAN TRAINING) ---
        # 1. One-Hot Encoding untuk fitur kategorikal
        # Kolom yang di-encode di training: 'fuel', 'seller_type', 'transmission', 'owner', 'name'
        cols_to_encode_for_pred = ['fuel', 'seller_type', 'transmission', 'owner', 'name']
        input_df_processed = pd.get_dummies(input_df, columns=cols_to_encode_for_pred, drop_first=True, dummy_na=False)

        # 2. Menyamakan Kolom dengan Kolom Training
        # Reindex input_df_processed agar memiliki kolom yang sama persis dengan X_train
        # Kolom yang tidak ada di input_df_processed (misal, kategori lain dari one-hot) akan diisi 0
        # Kolom yang ada di input_df_processed tapi tidak di training_columns akan di-drop
        input_df_aligned = input_df_processed.reindex(columns=training_columns, fill_value=0)
        
        # Pastikan tidak ada NaN setelah alignment (seharusnya tidak jika fill_value=0)
        if input_df_aligned.isnull().sum().sum() > 0:
            st.warning("Ditemukan NaN setelah penyesuaian kolom input. Mengisi dengan 0.")
            input_df_aligned = input_df_aligned.fillna(0)


        # --- MELAKUKAN PREDIKSI ---
        try:
            prediction = model.predict(input_df_aligned)
            predicted_price = prediction[0]
            st.success(f"Prediksi Harga Jual Mobil: **₹{predicted_price:,.0f}**")
            
            # Menampilkan detail input yang digunakan (opsional)
            with st.expander("Detail Input yang Digunakan untuk Prediksi"):
                st.write(input_df) # Tampilkan input sebelum di-encode agar mudah dibaca pengguna
                # st.write("Input setelah pra-pemrosesan (untuk debugging):")
                # st.dataframe(input_df_aligned)


        except Exception as e:
            st.error(f"Terjadi error saat melakukan prediksi: {e}")
            st.error("Pastikan semua input valid dan model telah dilatih dengan benar.")
            # st.dataframe(input_df_aligned) # Untuk debugging jika ada error dimensi

    # --- PENJELASAN TAMBAHAN MENGENAI RANDOM FOREST DAN OVERFITTING (DARI PERMINTAAN ANDA) ---
    st.markdown("---")
    st.header("Informasi Tambahan: Random Forest & Overfitting")

    with st.expander("1. Penanganan Overfitting pada Random Forest & Parameter"):
        st.markdown("""
        *Overfitting* terjadi ketika model terlalu mempelajari data training, termasuk noise-nya, sehingga performanya buruk pada data baru. Random Forest memiliki beberapa parameter untuk mengontrol kompleksitas dan mencegah overfitting:
        - **`n_estimators`**: Jumlah pohon. Lebih banyak pohon meningkatkan stabilitas, namun setelah titik tertentu hanya menambah waktu komputasi.
        - **`max_depth`**: Kedalaman maksimum setiap pohon. Membatasi `max_depth` (misalnya, 10, 15, 20) mencegah pohon menjadi terlalu kompleks. Jika `None`, pohon tumbuh hingga semua daun murni atau mencapai `min_samples_split`.
        - **`min_samples_split`**: Jumlah sampel minimum yang dibutuhkan untuk membagi (*split*) node internal. Nilai lebih tinggi membuat model lebih general.
        - **`min_samples_leaf`**: Jumlah sampel minimum yang harus ada di *leaf node*. Memastikan setiap prediksi akhir didasarkan pada sejumlah sampel yang cukup.
        - **`max_features`**: Jumlah fitur yang dipertimbangkan saat mencari *split* terbaik. Menguranginya (misal, `'sqrt'`, `'log2'`) membuat pohon lebih beragam dan mengurangi korelasi antar pohon, membantu mengatasi overfitting.

        Contoh `param_grid_rf` yang diberikan:
        ```
        param_grid_rf = {
            'n_estimators': ,
            'max_depth': [15, 20, None],
            'min_samples_split': [5][10],
            'min_samples_leaf': [2][4]
        }
        ```
        """)

    with st.expander("2. Parameter yang Paling Mungkin Menyebabkan Overfitting dari Pilihan Terbaik"):
        st.markdown("""
        Jika model terbaik yang dipilih memiliki `{'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100}`, maka parameter yang **paling mungkin menyebabkan overfitting** adalah **`'max_depth': None`**.
        Ini karena pohon diizinkan tumbuh tanpa batasan kedalaman eksplisit, berpotensi menangkap noise dalam data training, meskipun `min_samples_leaf` dan `min_samples_split` memberikan sedikit batasan.
        """)

    with st.expander("3. Parameter Lain yang Belum Dimasukkan dan Bisa Mengatasi Overfitting"):
        st.markdown("""
        Ya, ada parameter lain yang penting, dan salah satunya seringkali sudah dimasukkan dalam praktek yang baik:
        - **`max_features`**: Seperti dijelaskan di atas, ini sangat krusial. Jika belum ada di `param_grid_rf` Anda, sebaiknya ditambahkan (misalnya `['sqrt', 'log2', 0.7]`).
        - **`ccp_alpha` (Cost Complexity Pruning alpha)**: Parameter untuk *Minimal Cost-Complexity Pruning*. Nilai alpha yang lebih besar akan meningkatkan jumlah node yang dipangkas, menghasilkan model yang lebih sederhana dan kurang overfit.
        - Menggunakan **`oob_score=True`** saat inisialisasi `RandomForestRegressor` bukanlah parameter tuning, tetapi memberikan estimasi performa generalisasi tanpa set validasi terpisah, membantu mendeteksi overfitting selama pengembangan.
        """)
