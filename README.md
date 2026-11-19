# Perbandingan K-Means dan Hierarchical Clustering untuk Segmentasi Pelanggan

## 1. Topik penelitian

**Judul kerja:**

Analisis Perbandingan Algoritma K-Means dan Hierarchical Clustering untuk Segmentasi Pelanggan Berdasarkan Recency, Frequency, dan Monetary (RFM) pada Dataset Electronic Sales September 2023–September 2024.

Dataset penelitian adalah Electronic Sales Sep2023-Sep2024 dari Kaggle, bukan Olist.

## 2. Sumber dan struktur data

- Sumber: [Electronic Sales Sep2023-Sep2024 — Kaggle](https://www.kaggle.com/datasets/cameronseamons/electronic-sales-sep2023-sep2024)
- File: \`Electronic_sales_Sep2023-Sep2024.csv\`
- Periode yang tercantum: September 2023–September 2024.
- Kolom utama untuk RFM: \`Customer ID\`, \`Purchase Date\`, \`Order Status\`, \`Total Price\`, dan \`Add-on Total\`.

Jangan menganggap ukuran, jumlah pelanggan, atau kualitas data dari contoh/salinan dataset lain sebagai hasil final. Jalankan eksplorasi pada CSV yang benar-benar dipakai dan simpan hasil pemeriksaannya.

## 3. Definisi operasional RFM

| Fitur | Definisi dalam eksperimen | Kolom |
|---|---|---|
| Recency | Selisih hari antara tanggal acuan analisis dan tanggal pembelian Completed terakhir pelanggan. Semakin kecil berarti semakin baru. | Purchase Date |
| Frequency | Jumlah baris berstatus Completed per pelanggan. Ini mengasumsikan satu baris merepresentasikan satu pesanan. | Customer ID, Order Status |
| Monetary | Jumlah Total Price + Add-on Total pada pesanan Completed. | Total Price, Add-on Total |

Tanggal acuan default adalah satu hari setelah tanggal pembelian Completed paling akhir. Definisi Monetary harus diverifikasi terhadap dokumentasi dataset: jika Total Price sudah mencakup Add-on Total, jangan menjumlahkannya lagi. Jika tidak ada ID transaksi/order, asumsi satu baris = satu pesanan wajib dijelaskan sebagai keterbatasan.

## 4. Rancangan perbandingan yang adil

Kedua algoritma memakai:
- Pelanggan yang sama dan urutan data yang sama.
- Tiga fitur RFM yang sama.
- Transformasi dan standardisasi yang sama.
- Rentang jumlah cluster yang sama (k=2 sampai k=10, selama valid).
- Sampel evaluasi Silhouette yang sama, dipilih deterministik dengan random seed 42.
- Davies-Bouldin dihitung pada seluruh pelanggan.
- Lingkungan komputasi dan proses pencatatan waktu yang sama.

**Penting:** K-Means dan Hierarchical adalah metodologi yang berbeda. Jangan memaksa satu algoritma memakai konfigurasi internal algoritma lain. Setiap algoritma diuji pada seluruh rentang k, lalu k terbaik masing-masing dipilih secara mandiri berdasarkan Silhouette Score tertinggi. Selain itu, tabel perbandingan pada setiap k yang sama disediakan untuk perbandingan langsung *apple-to-apple*. Jika k terbaik keduanya berbeda, hal itu dilaporkan apa adanya.

## 5. Tahapan metodologi

1. **Eksplorasi data:** memeriksa ukuran data, tipe kolom, missing values, duplikasi, status pesanan, rentang tanggal, dan nilai transaksi.
2. **Pembersihan transaksi:** parsing tanggal dan nominal, mengeluarkan transaksi yang bukan Completed, dan memeriksa nilai yang tidak valid. Baris duplikat dilaporkan; jangan menghapus otomatis sebelum diketahui apakah itu duplikasi data atau transaksi yang sah.
3. **Pembentukan RFM:** mengagregasi Recency, Frequency, dan Monetary per Customer ID.
4. **Pemeriksaan RFM:** melihat statistik dan distribusi ketiga fitur sebelum clustering.
5. **Preprocessing bersama:** menerapkan \`log1p\` untuk mengurangi skewness, lalu \`StandardScaler\`. Transformasi dan scaler yang sama dipakai oleh dua algoritma. Outlier tidak otomatis dihapus.
6. **K-Means:** menguji k=2–10 dengan \`n_init=20\` dan \`random_state=42\`; menyimpan inertia, Silhouette, DBI, waktu fit, dan label setiap k. Elbow plot membantu membaca perubahan WCSS/inertia.
7. **Hierarchical:** membangun Agglomerative Hierarchical standar menggunakan Ward linkage dan Euclidean distance. Pohon hierarki dibangun satu kali dan dipotong pada k=2–10; dendrogram berasal dari pohon yang sama.
8. **Evaluasi:** Silhouette dihitung pada sampel deterministik maksimum 5.000 pelanggan yang sama untuk kedua metode; DBI dihitung pada seluruh pelanggan. Waktu K-Means adalah waktu fit per k. Waktu Hierarchical adalah waktu membangun pohon Ward satu kali, karena pohon tersebut digunakan untuk memotong beberapa nilai k.
9. **Pemilihan hasil:** k terbaik setiap algoritma ditentukan terpisah berdasarkan Silhouette tertinggi. Perbandingan semua nilai k tetap disediakan agar perbedaan k tidak disembunyikan.
10. **Interpretasi:** profil cluster dihitung memakai RFM asli (bukan data yang telah distandardisasi), disertai ukuran cluster dan proyeksi PCA 2D.

## 6. Algoritma dan parameter

### K-Means

K-Means membagi pelanggan ke dalam k cluster dengan meminimalkan jumlah kuadrat jarak anggota terhadap centroid cluster (within-cluster sum of squares / inertia).

Parameter:
- \`n_clusters\`: diuji 2 sampai 10.
- \`init\`: default scikit-learn, yaitu k-means++.
- \`n_init\`: 20 inisialisasi.
- \`random_state\`: 42.
- Jarak: Euclidean (sesuai implementasi K-Means).

### Agglomerative Hierarchical Clustering

Hierarchical bottom-up dimulai dari tiap pelanggan sebagai cluster sendiri, kemudian menggabungkan dua kelompok dengan kenaikan variasi dalam cluster terkecil sampai membentuk pohon hierarki. Pemotongan pohon menghasilkan label pada jumlah cluster yang diinginkan.

Parameter:
- \`linkage\`: Ward.
- \`metric\`: Euclidean.
- \`k\`: 2 sampai 10.
- Pohon hierarki dibangun sekali untuk konsistensi dendrogram dan efisiensi pemotongan beberapa k.

Ward linkage digunakan dengan Euclidean distance. Implementasi ini adalah Hierarchical standar tanpa graph tetangga (bukan connectivity-constrained variant). Karena kebutuhan memorinya meningkat kuadratik terhadap jumlah pelanggan, skrip memberi batas pengaman 12.000 pelanggan. Jika jumlah pelanggan melampaui batas, jangan diam-diam mengganti algoritma; evaluasi strategi komputasi/sampling bersama dosen pembimbing dan dokumentasikan perubahan metodologi.

## 7. Metrik evaluasi

- **Silhouette Score:** rentang -1 hingga 1; makin tinggi umumnya makin baik. Dihitung pada sampel tetap maksimum 5.000 pelanggan, sama untuk kedua algoritma.
- **Davies-Bouldin Index:** makin rendah umumnya makin baik. Dihitung pada seluruh pelanggan.
- **Inertia/WCSS:** hanya untuk K-Means; digunakan dalam Elbow Method.
- **Waktu fit:** K-Means dicatat per k. Hierarchical mencatat waktu membangun pohon penuh satu kali. Karena definisi waktunya berbeda secara struktural, jelaskan hal ini saat membahas efisiensi.
- **PCA 2D:** hanya proyeksi untuk visualisasi; PCA bukan input model clustering.

## 8. Struktur file program

| File | Fungsi |
|---|---|
| \`01_eksplorasi_dataset.py\` | Eksplorasi struktur dan kualitas CSV. |
| \`02_perhitungan_rfm.py\` | Menghitung RFM per pelanggan dan distribusinya. |
| \`03_preprocessing.py\` | Transformasi log1p, standardisasi, dan grafik sebelum/sesudah. |
| \`04_kmeans.py\` | Uji K-Means k=2–10, Elbow, Silhouette, inertia, waktu, dan label. |
| \`05_hierarchical.py\` | Pohon Ward, label per k, evaluasi, dan dendrogram. |
| \`06_evaluasi_perbandingan.py\` | Membandingkan semua k, menentukan k terbaik masing-masing, profil, dan grafik. |

## 9. Output visual dan tabel

- \`visualisasi_distribusi_rfm.png\`: distribusi Recency, Frequency, Monetary.
- \`visualisasi_preprocessing.png\`: distribusi sebelum dan sesudah log1p.
- \`grafik_elbow_kmeans.png\`: inertia terhadap k.
- \`grafik_silhouette_kmeans.png\`: Silhouette K-Means terhadap k.
- \`dendrogram_hierarchical.png\`: dendrogram pohon Ward yang sama dengan model.
- \`grafik_silhouette_perbandingan.png\`: Silhouette kedua algoritma pada setiap k.
- \`grafik_dbi_perbandingan.png\`: DBI kedua algoritma pada setiap k.
- \`grafik_waktu_perbandingan.png\`: waktu pembentukan model/pohon.
- \`scatter_kmeans.png\` dan \`scatter_hierarchical.png\`: titik pelanggan pada proyeksi PCA 2D, diwarnai menurut cluster terbaik masing-masing metode.
- \`grafik_ukuran_cluster.png\`: jumlah pelanggan pada setiap cluster.
- \`hasil_rfm.csv\`, \`rfm_scaled.csv\`, \`kmeans_metrics.csv\`, \`hierarchical_metrics.csv\`.
- \`hasil_kmeans.csv\`, \`hasil_hierarchical.csv\`: label final menurut k terbaik masing-masing algoritma.
- \`profil_cluster_kmeans.csv\`, \`profil_cluster_hierarchical.csv\`: karakteristik RFM asli setiap cluster.
- \`perbandingan_algoritma_semua_k.csv\`: perbandingan dua algoritma pada setiap k yang sama.
- \`perbandingan_k_terbaik.csv\`: hasil k terbaik masing-masing algoritma.
- \`ringkasan_perbandingan.txt\`: ringkasan angka dan interpretasi awal.

## 10. Cara menjalankan di Google Colab

Pastikan file CSV berada satu folder dengan seluruh skrip. Tarik pembaruan repository, lalu jalankan secara berurutan:

\`\`\`python
%cd /content/skripsi-RFM-Algoritma-K-Means-and-hirarical
!git pull origin main
!pip install -r requirements.txt

%run 01_eksplorasi_dataset.py
%run 02_perhitungan_rfm.py
%run 03_preprocessing.py
%run 04_kmeans.py
%run 05_hierarchical.py
%run 06_evaluasi_perbandingan.py
\`\`\`

Jangan menjalankan tahap berikutnya jika tahap sebelumnya gagal. Periksa output dan file CSV/PNG di panel Files Colab.

## 11. Batasan yang wajib diperiksa sebelum menulis kesimpulan

1. Pastikan satu baris CSV memang satu pesanan. Jika bukan, Frequency dan Monetary harus diagregasi berdasarkan ID pesanan yang benar.
2. Pastikan apakah \`Total Price\` sudah mencakup \`Add-on Total\`; hindari double counting.
3. Pastikan nilai status Completed, tanggal, nominal, dan duplikasi sesuai isi CSV aktual.
4. Full Ward linkage memerlukan memori kuadratik. Batas pengaman skrip adalah 12.000 pelanggan.
5. Hasil cluster bukan label kebenaran (ground truth). Penilaian perlu berdasarkan metrik internal, profil RFM, ukuran cluster, dan konteks.
6. Angka hasil penelitian harus berasal dari eksekusi terhadap CSV penelitian yang sebenarnya, bukan data sintetis pada pengujian otomatis.
