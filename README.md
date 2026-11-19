# Perbandingan K-Means dan Hierarchical Clustering untuk Segmentasi Pelanggan

## 1. Topik penelitian

**Judul kerja:**

Analisis Perbandingan Algoritma K-Means dan Hierarchical Clustering untuk Segmentasi Pelanggan Berdasarkan Recency, Frequency, dan Monetary (RFM) pada Dataset Electronic Sales September 2023–September 2024.

Dataset penelitian yang digunakan adalah Electronic Sales Sep2023-Sep2024 dari Kaggle, bukan Olist. Pastikan nama dataset konsisten di judul, latar belakang, sumber data, dan metodologi skripsi.

## 2. Sumber dan gambaran dataset

- Sumber: [Electronic Sales Sep2023-Sep2024 — Kaggle](https://www.kaggle.com/datasets/cameronseamons/electronic-sales-sep2023-sep2024)
- File: `Electronic_sales_Sep2023-Sep2024.csv`
- Periode: September 2023–September 2024.
- Kolom relevan: `Customer ID`, `Purchase Date`, `Order Status`, `Total Price`, dan `Add-on Total`.

Struktur dataset yang dipublikasikan pada salinan dataset yang dapat diperiksa menunjukkan 20.000 baris dan 16 kolom. Referensi proyek lain juga menjelaskan satu baris sebagai satu pesanan. Namun, skrip eksplorasi tetap harus dijalankan pada salinan CSV yang dipakai dalam penelitian ini untuk memastikan struktur, duplikasi, dan asumsi tersebut. Dataset tidak memiliki kolom Order ID khusus pada struktur yang digunakan.

## 3. Definisi operasional RFM

| Fitur | Definisi operasional | Kolom sumber |
|---|---|---|
| Recency | Jumlah hari dari tanggal acuan analisis ke tanggal pembelian Completed terakhir pelanggan. Semakin kecil semakin baru. | Purchase Date |
| Frequency | Jumlah baris pesanan berstatus Completed per pelanggan. Ini mengasumsikan satu baris = satu pesanan. | Customer ID, Order Status |
| Monetary | Total pengeluaran pesanan Completed, yaitu penjumlahan Total Price + Add-on Total. | Total Price, Add-on Total |

Tanggal acuan default adalah satu hari setelah tanggal pembelian Completed terakhir. `Total Price` dan `Add-on Total` diperlakukan sebagai dua komponen nilai pembelian yang berbeda; pastikan definisi ini sesuai dokumentasi dataset dan arahan dosen pembimbing.

## 4. Fokus penelitian dan batasan

Penelitian membandingkan K-Means dan Agglomerative Hierarchical Clustering menggunakan:
- Baris pelanggan dan fitur RFM yang sama.
- Prosedur pembersihan, transformasi, dan standardisasi yang sama.
- Jumlah cluster yang sama saat perbandingan akhir.
- Metrik evaluasi dan lingkungan komputasi yang sama.

Fitur demografis, jenis produk, metode pembayaran, dan rating tidak dipakai untuk membentuk cluster RFM. Fitur tersebut dapat dipakai sebagai konteks deskriptif tambahan jika diperlukan.

## 5. Metodologi eksperimen

1. Mengunduh dan mengeksplorasi dataset.
2. Memeriksa tipe data, missing values, duplikasi, status pesanan, tanggal, dan nilai transaksi.
3. Menggunakan pesanan Completed untuk membentuk RFM.
4. Mengagregasi data per Customer ID menjadi Recency, Frequency, dan Monetary.
5. Memeriksa distribusi fitur RFM.
6. Menerapkan transformasi `log1p` untuk mengurangi skewness, lalu `StandardScaler` agar skala ketiga fitur sebanding. Outlier tidak otomatis dihapus agar pelanggan bernilai tinggi tidak hilang dari populasi.
7. Menguji K-Means pada k=2 sampai k=10 (atau sampai k maksimum yang valid), lalu memilih k dengan Silhouette Score tertinggi pada rentang uji.
8. Menjalankan Agglomerative Hierarchical Clustering dengan linkage Ward dan Euclidean distance pada seluruh data, menggunakan graph tetangga terdekat untuk membatasi konektivitas.
9. Menggunakan nilai k yang sama dengan K-Means untuk perbandingan akhir.
10. Menghitung Silhouette Score, Davies-Bouldin Index, dan waktu fit.
11. Menginterpretasikan cluster menggunakan nilai RFM asli, bukan nilai standardisasi.
12. Menyimpulkan hasil dengan mempertimbangkan metrik, profil, ukuran cluster, dan keterbatasan metode.

**Catatan dendrogram:** model Hierarchical dan metriknya dihitung pada seluruh pelanggan. Dendrogram divisualisasikan dari sampel pelanggan yang dipilih secara deterministik (maksimum 1.000) agar grafik dapat dibaca dan komputasi visualisasi lebih ringan. Karena itu, dendrogram adalah ilustrasi struktur penggabungan pada sampel, bukan pohon lengkap seluruh pelanggan.

## 6. Metrik evaluasi

- **Silhouette Score:** rentang -1 sampai 1; semakin tinggi umumnya semakin baik.
- **Davies-Bouldin Index (DBI):** semakin rendah umumnya semakin baik.
- **Waktu fit:** durasi proses pelatihan/fitting algoritma. Waktu preprocessing dan evaluasi tidak dicampurkan ke angka ini.

Label cluster seperti 0, 1, dan 2 adalah penanda teknis, bukan urutan kualitas atau ranking pelanggan.

## 7. Struktur file

| File | Fungsi |
|---|---|
| `01_eksplorasi_dataset.py` | Membaca dataset dan memeriksa struktur, missing values, status, serta statistik. |
| `02_perhitungan_rfm.py` | Menghitung RFM per pelanggan dan menyimpan `hasil_rfm.csv`. |
| `03_preprocessing.py` | Transformasi log1p, standardisasi, dan visualisasi distribusi preprocessing. |
| `04_kmeans.py` | Elbow, Silhouette Score untuk beberapa k, hasil cluster, profil, dan scatter plot PCA. |
| `05_hierarchical.py` | Agglomerative Hierarchical, dendrogram sampel, profil cluster, dan scatter plot PCA. |
| `06_evaluasi_perbandingan.py` | Tabel dan grafik perbandingan metrik serta ringkasan hasil. |

File hasil yang dibuat otomatis:
- `hasil_rfm.csv`
- `visualisasi_distribusi_rfm.png`
- `rfm_scaled.csv`
- `preprocessing_info.json`
- `visualisasi_preprocessing.png`
- `kmeans_metrics.csv`
- `hasil_kmeans.csv`
- `profil_cluster_kmeans.csv`
- `grafik_elbow_kmeans.png`
- `grafik_silhouette_kmeans.png`
- `scatter_kmeans.png`
- `hierarchical_metrics.csv`
- `hasil_hierarchical.csv`
- `profil_cluster_hierarchical.csv`
- `dendrogram_hierarchical.png`
- `scatter_hierarchical.png`
- `perbandingan_algoritma.csv`
- `grafik_perbandingan_algoritma.png`
- `ringkasan_perbandingan.txt`

## 8. Cara menjalankan di Google Colab

Pastikan file CSV berada satu folder dengan seluruh file Python. Setelah `git pull origin main`, jalankan satu per satu dalam notebook:

```python
%run 01_eksplorasi_dataset.py
%run 02_perhitungan_rfm.py
%run 03_preprocessing.py
%run 04_kmeans.py
%run 05_hierarchical.py
%run 06_evaluasi_perbandingan.py
```

Jika library belum tersedia:

```python
!pip install -r requirements.txt
```

Jalankan berurutan karena setiap skrip membaca hasil dari tahap sebelumnya.

## 9. Catatan metodologis dan batasan

- Pastikan asumsi satu baris = satu pesanan benar untuk file CSV yang dianalisis. Jika tidak, Frequency dan Monetary harus dihitung setelah agregasi berdasarkan ID pesanan yang benar.
- Periksa definisi Total Price dan Add-on Total agar tidak terjadi penghitungan ganda.
- Standardisasi harus memakai fitur RFM yang sama untuk kedua algoritma.
- Agglomerative Hierarchical dapat lebih berat daripada K-Means pada jumlah pelanggan besar. Penggunaan graph tetangga membantu membatasi konektivitas, tetapi waktu dan kebutuhan memori tetap perlu diamati saat eksperimen.
- Dendrogram yang ditampilkan adalah sampel, sedangkan hasil cluster dan evaluasi model menggunakan seluruh pelanggan.
- Hasil clustering tidak otomatis menjadi segmentasi pemasaran yang benar. Interpretasi perlu didukung profil RFM dan keterbatasan dataset.
- Hasil akhir dan angka penelitian hanya boleh ditulis setelah skrip dijalankan pada CSV penelitian dan outputnya diperiksa.
