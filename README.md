# 📊 Telco Customer Churn Prediction: Multi-Model Analysis & Interpretability
> **Domain:** Business Intelligence & Telecommunication | **Tech Stack:** Python, Scikit-Learn, Machine Learning

## 📌 Ringkasan Proyek
Proyek ini bertujuan untuk membangun model prediktif yang mampu mengidentifikasi pelanggan yang berisiko berhenti berlangganan (*churn*) pada perusahaan telekomunikasi[cite: 2]. Dengan menggunakan dataset yang mencakup 7.043 pelanggan, proyek ini mensimulasikan siklus hidup proyek *Data Science* secara *end-to-end*, mulai dari pembersihan data kotor hingga analisis strategi retensi pelanggan berdasarkan hasil model[cite: 2].

## 🛠️ Tech Stack & Library
*   **Bahasa:** Python 3.11[cite: 2]
*   **Manipulasi Data:** Pandas, NumPy[cite: 2]
*   **Visualisasi:** Matplotlib, Seaborn[cite: 2]
*   **Machine Learning:** Scikit-Learn (Logistic Regression, Decision Tree, KNN)[cite: 2]
*   **Interpretability:** SHAP & Decision Rules Visualization[cite: 2]

## 🚀 Alur Kerja Proyek

### 1. Data Understanding & Cleaning
*   **Penanganan Anomali:** Mengidentifikasi dan memperbaiki kolom `TotalCharges` yang terbaca sebagai *object* (string) karena adanya karakter spasi kosong pada pelanggan baru (*tenure* 0)[cite: 2].
*   **Imputasi:** Melakukan konversi tipe data ke numerik dan mengisi *missing values* dengan nilai 0 berdasarkan logika bisnis[cite: 2].

### 2. Exploratory Data Analysis (EDA)
*   **Class Imbalance:** Menemukan bahwa target bersifat tidak seimbang dengan rasio 73.5% (Setia) berbanding 26.5% (Churn)[cite: 2].
*   **Insight Utama:** Pelanggan dengan kontrak bulanan (*month-to-month*) dan pengguna layanan *Fiber Optic* memiliki probabilitas *churn* yang jauh lebih tinggi dibandingkan segmen lainnya[cite: 2].

### 3. Data Preprocessing
*   **Encoding:** Menggunakan *One-Hot Encoding* dengan `drop_first=True` untuk mengubah 20 fitur kategorikal menjadi 31 fitur numerik tanpa terjebak dalam *Dummy Variable Trap*[cite: 2].
*   **Scaling:** Menerapkan `StandardScaler` (Z-Score Normalization) untuk menyeragamkan skala fitur numerik guna mengoptimalkan algoritma berbasis jarak (KNN) dan gradien (Logistic Regression)[cite: 2].
*   **Stratified Splitting:** Membagi data (80:20) dengan menjaga proporsi kelas target[cite: 2].

### 4. Modeling & Optimization
Mengevaluasi tiga algoritma berbeda dengan optimasi hiperparameter:
*   **K-Nearest Neighbors (KNN):** Menggunakan `GridSearchCV` untuk menemukan nilai $K$ optimal ($K=21$)[cite: 2].
*   **Decision Tree:** Membatasi `max_depth` untuk mencegah *overfitting*[cite: 2].
*   **Logistic Regression:** Sebagai model *baseline* linear yang robust[cite: 2].

## 📈 Hasil Evaluasi Model

| Model | Test Accuracy | CV Mean Accuracy | AUC-ROC Score |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | **80.70%** | **80.25%** | **0.8418** |
| Decision Tree | 79.42% | 78.97% | 0.8267 |
| KNN (Tuned) | 77.08% | 78.83% | 0.8105 |

**Kesimpulan:** Logistic Regression memberikan performa terbaik dalam membedakan pelanggan yang akan *churn* dengan nilai AUC tertinggi (0.8418)[cite: 2].

## 💡 Interpretabilitas & Rekomendasi Bisnis
Berdasarkan analisis *Feature Importance* dan *Decision Rules*:
1.  **Faktor Risiko:** Tagihan bulanan yang tinggi dan penggunaan layanan *Fiber Optic* adalah pendorong utama pelanggan untuk berhenti[cite: 2].
2.  **Faktor Retensi:** Durasi berlangganan (*tenure*) yang lama dan kontrak jangka panjang (1-2 tahun) sangat efektif dalam menjaga loyalitas pelanggan[cite: 2].
3.  **Strategi:** Perusahaan disarankan memberikan promo loyalitas pada bulan-bulan awal kontrak untuk meningkatkan *tenure* pelanggan baru[cite: 2].

## 📁 Struktur Repositori
```text
.
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset mentah
├── RC_AIProject.ipynb                    # Notebook analisis lengkap
└── README.md                             # Dokumentasi proyek
```
