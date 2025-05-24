# train_model.py
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib # Untuk menyimpan model dan objek lainnya
import pickle # Alternatif atau pelengkap untuk joblib

def train_and_save_model():
    print("Memulai proses pelatihan model...")
    # --- 1. MEMUAT DATA ---
    try:
        df_raw = pd.read_csv('cardekho.csv')
        print("Dataset cardekho.csv berhasil dimuat.")
    except FileNotFoundError:
        print("Error: cardekho.csv tidak ditemukan. Pastikan file ada di direktori yang sama.")
        return

    df = df_raw.copy()

    # --- 2. PRA-PEMROSESAN AWAL (Sama seperti di notebook Anda) ---
    # Hapus baris dengan NaN krusial jika ada (contoh dari notebook asli)
    # Ini mungkin berbeda dari dataset cardekho, sesuaikan jika perlu
    # df.dropna(subset=['mileage(km/ltr/kg)', 'engine', 'max_power', 'seats'], inplace=True)

    # Feature Engineering: Usia Mobil (car_age)
    if 'year' in df.columns:
        current_year = datetime.now().year
        df['car_age'] = current_year - df['year']
    else:
        print("Peringatan: Kolom 'year' tidak ditemukan untuk membuat 'car_age'.")


    # Pembersihan dan Konversi Kolom Numerik
    raw_numeric_cols_to_process = ['mileage(km/ltr/kg)', 'engine', 'max_power', 'seats']
    for col_name in raw_numeric_cols_to_process:
        if col_name in df.columns:
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
            if df[col_name].isnull().sum() > 0:
                median_val = df[col_name].median()
                df[col_name].fillna(median_val, inplace=True)
    print("Pembersihan kolom numerik dan pengisian NaN selesai.")

    # Standarisasi Nama Kolom
    rename_map = {'mileage(km/ltr/kg)': 'mileage'}
    for original_name, new_name in rename_map.items():
        if original_name in df.columns and original_name != new_name:
            df.rename(columns={original_name: new_name}, inplace=True)
    print("Standarisasi nama kolom selesai.")

    # Menyimpan opsi untuk dropdown di aplikasi Streamlit
    # Ambil nilai unik SEBELUM one-hot encoding untuk categorical_cols
    input_options = {}
    categorical_cols_for_input = ['fuel', 'seller_type', 'transmission', 'owner', 'name'] # 'name' untuk car model
    for col in categorical_cols_for_input:
        if col in df.columns:
            # Urutkan untuk konsistensi dropdown, kecuali 'name' yang bisa sangat banyak
            unique_values = sorted(df[col].dropna().unique().tolist()) if col != 'name' else df[col].dropna().unique().tolist()
            input_options[col] = unique_values


    # Encoding Fitur Kategorikal
    categorical_cols_to_encode = ['fuel', 'seller_type', 'transmission', 'owner', 'name'] # Termasuk 'name'
    existing_categorical_cols = [col for col in categorical_cols_to_encode if col in df.columns]
    if existing_categorical_cols:
        df = pd.get_dummies(df, columns=existing_categorical_cols, drop_first=True, dummy_na=False) # dummy_na=False agar konsisten
    print("Encoding fitur kategorikal selesai.")


    # Menghapus kolom yang tidak relevan atau sudah di-engineer
    cols_to_drop_final = []
    if 'car_age' in df.columns and 'year' in df.columns:
        cols_to_drop_final.append('year')
    # 'name' sudah di-encode, jadi nama asli tidak perlu di-drop secara eksplisit lagi jika sudah jadi dummy
    # Kolom lain yang mungkin perlu di-drop, misalnya kolom ID jika ada.

    existing_cols_to_drop_final = [col for col in cols_to_drop_final if col in df.columns]
    if existing_cols_to_drop_final:
        df.drop(columns=existing_cols_to_drop_final, inplace=True, errors='ignore')
    print("Penghapusan kolom tidak relevan selesai.")

    # Seleksi Fitur (X) dan Target (y)
    if 'selling_price' not in df.columns:
        print("Error: Kolom target 'selling_price' tidak ditemukan. Pelatihan dihentikan.")
        return

    # Drop baris di mana target adalah NaN (jika ada)
    df.dropna(subset=['selling_price'], inplace=True)

    X = df.drop('selling_price', axis=1)
    y = df['selling_price']

    # Pastikan tidak ada NaN di X setelah semua proses
    if X.isnull().sum().sum() > 0:
        print(f"Peringatan: Ditemukan {X.isnull().sum().sum()} NaN di X sebelum training. Mengisi dengan 0...")
        X = X.fillna(0) # Strategi fallback sederhana, idealnya ditangani lebih baik

    print(f"Shape X: {X.shape}, Shape y: {y.shape}")

    # --- 3. PEMBAGIAN DATA ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data berhasil dibagi menjadi set training dan testing.")

    # --- 4. PELATIHAN MODEL DENGAN GRIDSEARCHCV ---
    print("Memulai pelatihan Random Forest dengan GridSearchCV...")
    param_grid_rf = {
        'n_estimators': [100, 150],
        'max_depth': [15, 20, None],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4],
        'max_features': ['sqrt', 'log2']
    }
    rf_model = RandomForestRegressor(random_state=42, oob_score=True)
    grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf,
                                  cv=3, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')
    grid_search_rf.fit(X_train, y_train)

    best_rf_model = grid_search_rf.best_estimator_
    print(f"Parameter terbaik ditemukan: {grid_search_rf.best_params_}")
    print(f"OOB Score dari model terbaik: {best_rf_model.oob_score_:.4f}")

    # --- 5. MENYIMPAN MODEL, KOLOM TRAINING, DAN OPSI INPUT ---
    model_filename = 'model.joblib'
    joblib.dump(best_rf_model, model_filename)
    print(f"Model berhasil disimpan sebagai {model_filename}")

    training_cols_filename = 'training_columns.pkl'
    with open(training_cols_filename, 'wb') as f:
        pickle.dump(X_train.columns.tolist(), f) # Simpan daftar nama kolom dari X_train
    print(f"Kolom training berhasil disimpan sebagai {training_cols_filename}")

    input_options_filename = 'input_options.pkl'
    with open(input_options_filename, 'wb') as f:
        pickle.dump(input_options, f)
    print(f"Opsi input berhasil disimpan sebagai {input_options_filename}")

    print("Proses pelatihan dan penyimpanan model selesai.")

if __name__ == '__main__':
    train_and_save_model()
