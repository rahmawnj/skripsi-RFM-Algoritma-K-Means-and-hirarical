# Perbandingan K-Means dan Hierarchical Clustering untuk Segmentasi Pelanggan

## 1. Topik penelitian

**Judul kerja yang disesuaikan dengan dataset:**

Analisis Perbandingan Algoritma K-Means dan Hierarchical Clustering untuk Segmentasi Pelanggan Berdasarkan Recency, Frequency, dan Monetary (RFM) pada Dataset Electronic Sales September 2023–September 2024.

> Proposal awal menyebut dataset Olist, tetapi dataset yang benar-benar digunakan dalam proyek ini adalah Electronic Sales Sep2023-Sep2024 dari Kaggle. Nama Olist perlu dihapus dari judul, latar belakang, sumber data, dan metodologi agar konsisten.

## 2. Sumber dan gambaran dataset

- Sumber: [Electronic Sales Sep2023-Sep2024 — Kaggle](https://www.kaggle.com/datasets/cameronseamons/electronic-sales-sep2023-sep2024)
- File: Electronic_sales_Sep2023-Sep2024.csv
- Periode data: September 2023–September 2024.
- Dataset berisi catatan pembelian pelanggan pada bisnis penjualan produk elektronik.
- Kolom yang relevan untuk RFM:
  - Customer ID: identitas pelanggan.
  - Purchase Date: tanggal pembelian.
  - Order Status: status pesanan, misalnya Completed atau Cancelled.
  - Total Price: nilai harga produk dalam pesanan.
  - Add-on Total: nilai pembelian add-on.

Dataset tidak menyediakan kolom Order ID/Transaction ID pada struktur yang digunakan. Untuk eksperimen ini, satu baris diperlakukan sebagai satu pesanan sesuai deskripsi struktur dataset. Asumsi ini perlu ditulis secara eksplisit dan diverifikasi kembali melalui eksplorasi data.

## 3. Fokus dan batasan penelitian

Penelitian berfokus pada segmentasi pelanggan berdasarkan perilaku pembelian yang tercatat dalam dataset Electronic Sales. Pesanan dengan status selain Completed tidak dihitung sebagai pembelian terealisasi dalam pembentukan RFM.

K-Means dan Agglomerative Hierarchical Clustering dibandingkan menggunakan:
- Dataset dan baris pelanggan yang sama.
- Fitur RFM yang sama.
- Tahapan pembersihan dan standardisasi yang sama.
- Jumlah cluster yang sebanding.
- Metrik evaluasi dan lingkungan komputasi yang sama.

Fitur demografis, jenis produk, metode pembayaran, dan rating tidak digunakan sebagai fitur pembentuk cluster utama. Informasi tersebut dapat digunakan untuk membantu mendeskripsikan profil segmen, bukan untuk membentuk cluster RFM.

## 4. Definisi operasional RFM

| Fitur | Definisi operasional | Kolom sumber |
|---|---|---|
| Recency | Jumlah hari antara tanggal acuan analisis dan tanggal pembelian Completed terakhir pelanggan. Semakin kecil nilainya, semakin baru pembeliannya. | Purchase Date |
| Frequency | Jumlah pesanan berstatus Completed milik pelanggan. Karena tidak tersedia Order ID, satu baris diperlakukan sebagai satu pesanan. | Customer ID, Order Status |
| Monetary | Total pengeluaran pembelian pelanggan, dihitung dari penjumlahan Total Price dan Add-on Total pada pesanan Completed. | Total Price, Add-on Total |

Tanggal acuan default ditetapkan satu hari setelah tanggal pembelian Completed paling akhir dalam dataset. Tanggal ini harus dilaporkan agar perhitungan Recency dapat direplikasi.

## 5. Rumusan masalah

1. Bagaimana penerapan K-Means dan Agglomerative Hierarchical Clustering untuk segmentasi pelanggan berdasarkan fitur RFM pada dataset Electronic Sales?
2. Bagaimana karakteristik setiap segmen pelanggan berdasarkan Recency, Frequency, dan Monetary?
3. Bagaimana perbandingan kualitas cluster dan waktu komputasi kedua algoritma berdasarkan Silhouette Score dan Davies–Bouldin Index?

## 6. Tujuan penelitian

1. Menerapkan K-Means dan Agglomerative Hierarchical Clustering pada fitur RFM pelanggan.
2. Mengidentifikasi dan menginterpretasikan karakteristik setiap segmen pelanggan.
3. Membandingkan kualitas hasil clustering dan waktu komputasi kedua algoritma.

## 7. Metodologi eksperimen

1. Studi literatur tentang RFM, K-Means, Hierarchical Clustering, standardisasi, dan evaluasi clustering.
2. Mengunduh dan memeriksa dataset Electronic Sales.
3. Membersihkan data: memeriksa nilai kosong, tipe data, duplikasi, tanggal, nominal, dan status pesanan.
4. Memilih pesanan Completed untuk menghitung pembelian yang terealisasi.
5. Mengagregasi data per Customer ID menjadi Recency, Frequency, dan Monetary.
6. Memeriksa distribusi dan pencilan fitur RFM.
7. Melakukan transformasi yang diperlukan dan standardisasi fitur dengan metode yang sama untuk kedua algoritma.
8. Menentukan jumlah cluster menggunakan prosedur konsisten, misalnya pengujian beberapa nilai k dan analisis pendukung.
9. Menjalankan K-Means dan Agglomerative Hierarchical Clustering pada data RFM terstandardisasi.
10. Mengevaluasi hasil menggunakan Silhouette Score, Davies–Bouldin Index, dan waktu komputasi.
11. Menginterpretasikan profil cluster menggunakan nilai RFM asli, bukan hanya nilai hasil standardisasi.
12. Menarik kesimpulan berdasarkan hasil eksperimen.

## 8. Metrik evaluasi

- Silhouette Score: semakin tinggi umumnya semakin baik, dengan rentang -1 sampai 1.
- Davies–Bouldin Index (DBI): semakin rendah umumnya semakin baik.
- Waktu komputasi: durasi proses clustering, diukur dengan prosedur dan perangkat yang sama.

Catat jumlah cluster, jumlah pelanggan yang dianalisis, parameter algoritma, linkage dan metric Hierarchical Clustering, serta random seed K-Means.

## 9. Struktur file proyek

- 01_eksplorasi_dataset.py — membaca dan mengeksplorasi CSV secara otomatis.
- 02_perhitungan_rfm.py — membentuk tabel RFM dan menyimpan hasil_rfm.csv.
- Electronic_sales_Sep2023-Sep2024.csv — dataset penelitian (file lokal; tidak harus di-commit ke GitHub).
- hasil_rfm.csv — hasil olahan yang dihasilkan oleh script.

## 10. Catatan metodologis penting

- Jangan menyebut dataset ini sebagai Olist. Dataset Olist dan Electronic Sales adalah dua dataset berbeda.
- Pastikan asumsi satu baris = satu pesanan sesuai dengan struktur file yang benar-benar dianalisis. Jika satu pesanan ternyata tersebar dalam beberapa baris, Frequency dan Monetary harus diagregasi per pesanan terlebih dahulu.
- Monetary menggunakan Total Price + Add-on Total sebagai definisi total pengeluaran. Keputusan ini perlu dijelaskan dan dikonfirmasi dengan dosen pembimbing. Jika hanya nilai produk yang dikehendaki, gunakan Total Price saja.
- Standardisasi harus diterapkan pada matriks RFM yang sama sebelum data diberikan kepada kedua algoritma.
- Hasil clustering tidak otomatis menunjukkan segmen pemasaran yang benar; interpretasi harus didukung profil RFM dan keterbatasan dataset.
