# 06_evaluasi_perbandingan.py
# Evaluasi dua metodologi secara terpisah dan perbandingan langsung pada setiap k yang sama.
# Setiap algoritma mempertahankan k terbaiknya sendiri berdasarkan Silhouette Score.

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

SCRIPT_DIR = Path(__file__).resolve().parent
RFM_PATH = SCRIPT_DIR / "hasil_rfm.csv"
SCALED_PATH = SCRIPT_DIR / "rfm_scaled.csv"
KMEANS_METRICS_PATH = SCRIPT_DIR / "kmeans_metrics.csv"
HIERARCHICAL_METRICS_PATH = SCRIPT_DIR / "hierarchical_metrics.csv"
KMEANS_LABELS_ALL_PATH = SCRIPT_DIR / "hasil_kmeans_semua_k.csv"
HIERARCHICAL_LABELS_ALL_PATH = SCRIPT_DIR / "hasil_hierarchical_semua_k.csv"

COMPARISON_PATH = SCRIPT_DIR / "perbandingan_algoritma_semua_k.csv"
BEST_COMPARISON_PATH = SCRIPT_DIR / "perbandingan_k_terbaik.csv"
KMEANS_LABELS_PATH = SCRIPT_DIR / "hasil_kmeans.csv"
HIERARCHICAL_LABELS_PATH = SCRIPT_DIR / "hasil_hierarchical.csv"
KMEANS_PROFILE_PATH = SCRIPT_DIR / "profil_cluster_kmeans.csv"
HIERARCHICAL_PROFILE_PATH = SCRIPT_DIR / "profil_cluster_hierarchical.csv"
SCORE_CHART_PATH = SCRIPT_DIR / "grafik_silhouette_perbandingan.png"
DBI_CHART_PATH = SCRIPT_DIR / "grafik_dbi_perbandingan.png"
TIME_CHART_PATH = SCRIPT_DIR / "grafik_waktu_perbandingan.png"
KMEANS_SCATTER_PATH = SCRIPT_DIR / "scatter_kmeans.png"
HIERARCHICAL_SCATTER_PATH = SCRIPT_DIR / "scatter_hierarchical.png"
SIZE_CHART_PATH = SCRIPT_DIR / "grafik_ukuran_cluster.png"
REPORT_PATH = SCRIPT_DIR / "ringkasan_perbandingan.txt"

SCALED_COLUMNS = ["R_scaled", "F_scaled", "M_scaled"]
RANDOM_STATE = 42

def build_profile(result):
    return result.groupby("Cluster").agg(
        Jumlah_Pelanggan=("Customer", "count"),
        Rata_Rata_Recency=("Recency", "mean"),
        Median_Recency=("Recency", "median"),
        Rata_Rata_Frequency=("Frequency", "mean"),
        Median_Frequency=("Frequency", "median"),
        Rata_Rata_Monetary=("Monetary", "mean"),
        Median_Monetary=("Monetary", "median"),
    ).round(2)

def make_scatter(X, labels, title, output_path):
    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
    fig, ax = plt.subplots(figsize=(9, 7))
    points = ax.scatter(
        coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=14, alpha=0.7
    )
    ax.set_title(title)
    ax.set_xlabel("Komponen Utama 1")
    ax.set_ylabel("Komponen Utama 2")
    fig.colorbar(points, ax=ax, label="Cluster")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.show()

def main():
    required = [
        RFM_PATH, SCALED_PATH, KMEANS_METRICS_PATH, HIERARCHICAL_METRICS_PATH,
        KMEANS_LABELS_ALL_PATH, HIERARCHICAL_LABELS_ALL_PATH,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"{path.name} belum tersedia. Jalankan seluruh tahap sebelumnya.")

    original = pd.read_csv(RFM_PATH)
    scaled = pd.read_csv(SCALED_PATH)
    km_metrics = pd.read_csv(KMEANS_METRICS_PATH)
    hc_metrics = pd.read_csv(HIERARCHICAL_METRICS_PATH)
    km_all = pd.read_csv(KMEANS_LABELS_ALL_PATH)
    hc_all = pd.read_csv(HIERARCHICAL_LABELS_ALL_PATH)

    if not (len(original) == len(scaled) == len(km_all) == len(hc_all)):
        raise ValueError("Jumlah baris antarfile tidak sama. Ulangi preprocessing dan clustering.")
    customer_ids = original["Customer"].astype(str)
    for frame, name in [(scaled, "preprocessing"), (km_all, "K-Means"), (hc_all, "Hierarchical")]:
        if not customer_ids.equals(frame["Customer"].astype(str)):
            raise ValueError(f"Urutan Customer pada file {name} berbeda.")

    # Gabungkan hasil untuk semua k agar perbandingan tidak bergantung pada satu pilihan k.
    km = km_metrics[["k", "silhouette_score", "davies_bouldin_index", "fit_time_seconds"]].copy()
    km["Algoritma"] = "K-Means"
    km = km.rename(columns={
        "silhouette_score": "Silhouette_Score",
        "davies_bouldin_index": "Davies_Bouldin_Index",
        "fit_time_seconds": "Waktu_Fit_Detik",
    })

    hc = hc_metrics[["k", "silhouette_score", "davies_bouldin_index", "tree_fit_time_seconds"]].copy()
    hc["Algoritma"] = "Agglomerative Hierarchical (Ward)"
    hc = hc.rename(columns={
        "silhouette_score": "Silhouette_Score",
        "davies_bouldin_index": "Davies_Bouldin_Index",
        "tree_fit_time_seconds": "Waktu_Fit_Detik",
    })

    comparison_all = pd.concat([km, hc], ignore_index=True)
    comparison_all = comparison_all.sort_values(["k", "Algoritma"])
    comparison_all.to_csv(COMPARISON_PATH, index=False)

    # Pilihan k terbaik setiap algoritma dilaporkan secara mandiri.
    best_km_row = km_metrics.loc[km_metrics["silhouette_score"].idxmax()]
    best_hc_row = hc_metrics.loc[hc_metrics["silhouette_score"].idxmax()]
    best_km = int(best_km_row["k"])
    best_hc = int(best_hc_row["k"])

    best_comparison = pd.DataFrame([
        {
            "Algoritma": "K-Means",
            "k_terbaik": best_km,
            "Silhouette_Score": float(best_km_row["silhouette_score"]),
            "Davies_Bouldin_Index": float(best_km_row["davies_bouldin_index"]),
            "Waktu_Fit_Detik": float(best_km_row["fit_time_seconds"]),
            "Dasar_pemilihan": "Silhouette Score tertinggi K-Means pada rentang k yang diuji",
        },
        {
            "Algoritma": "Agglomerative Hierarchical (Ward)",
            "k_terbaik": best_hc,
            "Silhouette_Score": float(best_hc_row["silhouette_score"]),
            "Davies_Bouldin_Index": float(best_hc_row["davies_bouldin_index"]),
            "Waktu_Fit_Detik": float(best_hc_row["tree_fit_time_seconds"]),
            "Dasar_pemilihan": "Silhouette Score tertinggi Hierarchical pada rentang k yang diuji",
        },
    ])
    best_comparison.to_csv(BEST_COMPARISON_PATH, index=False)

    # Ambil label final dari k terbaik masing-masing metode.
    km_labels = km_all[f"Cluster_k{best_km}"].to_numpy()
    hc_labels = hc_all[f"Cluster_k{best_hc}"].to_numpy()

    km_result = original.copy()
    km_result["Cluster"] = km_labels
    hc_result = original.copy()
    hc_result["Cluster"] = hc_labels
    km_result.to_csv(KMEANS_LABELS_PATH, index=False)
    hc_result.to_csv(HIERARCHICAL_LABELS_PATH, index=False)

    km_profile = build_profile(km_result)
    hc_profile = build_profile(hc_result)
    km_profile.to_csv(KMEANS_PROFILE_PATH)
    hc_profile.to_csv(HIERARCHICAL_PROFILE_PATH)

    # Grafik perbandingan metrik pada setiap nilai k yang sama.
    metric_specs = [
        ("Silhouette_Score", "Silhouette Score (lebih tinggi lebih baik)", SCORE_CHART_PATH),
        ("Davies_Bouldin_Index", "Davies-Bouldin Index (lebih rendah lebih baik)", DBI_CHART_PATH),
        ("Waktu_Fit_Detik", "Waktu pembentukan model/pohon (detik)", TIME_CHART_PATH),
    ]
    for column, title, output_path in metric_specs:
        pivot = comparison_all.pivot(index="k", columns="Algoritma", values=column)
        ax = pivot.plot(marker="o", figsize=(9, 5))
        ax.set_title(title)
        ax.set_xlabel("Jumlah cluster (k)")
        ax.set_ylabel(column)
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.show()

    X = scaled[SCALED_COLUMNS].to_numpy()
    make_scatter(X, km_labels, f"K-Means - PCA 2D (k terbaik={best_km})", KMEANS_SCATTER_PATH)
    make_scatter(
        X, hc_labels,
        f"Hierarchical Ward - PCA 2D (k terbaik={best_hc})",
        HIERARCHICAL_SCATTER_PATH,
    )

    # Distribusi jumlah pelanggan pada k terbaik masing-masing metode.
    size_data = pd.concat([
        km_result.assign(Algoritma=f"K-Means (k={best_km})")[["Customer", "Cluster", "Algoritma"]],
        hc_result.assign(Algoritma=f"Hierarchical (k={best_hc})")[["Customer", "Cluster", "Algoritma"]],
    ], ignore_index=True)
    size_table = size_data.groupby(["Algoritma", "Cluster"]).size().unstack(fill_value=0)
    ax = size_table.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Jumlah Pelanggan per Cluster pada k Terbaik Masing-Masing Metode")
    ax.set_xlabel("Algoritma")
    ax.set_ylabel("Jumlah pelanggan")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(SIZE_CHART_PATH, dpi=180, bbox_inches="tight")
    plt.show()

    report_lines = [
        "RINGKASAN PERBANDINGAN K-MEANS DAN HIERARCHICAL",
        "=" * 58,
        f"Jumlah pelanggan dianalisis: {len(original):,}",
        "",
        "METODOLOGI PEMILIHAN JUMLAH CLUSTER",
        "- K-Means dan Hierarchical diuji pada rentang k yang sama.",
        "- k terbaik setiap algoritma dipilih secara mandiri berdasarkan Silhouette Score tertinggi.",
        "- Perbandingan langsung juga disediakan pada setiap k yang sama di perbandingan_algoritma_semua_k.csv.",
        "- Karena k terbaik dapat berbeda, hasil terbaik masing-masing metode tidak dianggap sebagai perbandingan pada k identik.",
        "",
        "HASIL K TERBAIK MASING-MASING ALGORITMA",
        best_comparison.to_string(index=False),
        "",
        "HASIL SEMUA K",
        comparison_all.to_string(index=False),
        "",
        "PROFIL CLUSTER K-MEANS (RFM ASLI)",
        km_profile.to_string(),
        "",
        "PROFIL CLUSTER HIERARCHICAL (RFM ASLI)",
        hc_profile.to_string(),
        "",
        "CATATAN INTERPRETASI",
        "- Silhouette lebih tinggi umumnya menunjukkan cluster lebih kompak dan terpisah.",
        "- Davies-Bouldin lebih rendah umumnya menunjukkan cluster lebih baik.",
        "- K-Means membentuk partisi dengan meminimalkan within-cluster sum of squares.",
        "- Hierarchical Ward menggabungkan kelompok secara bertahap dengan kenaikan within-cluster variance terkecil.",
        "- Waktu Hierarchical adalah waktu membangun pohon Ward sekali; pohon yang sama dipotong untuk beberapa nilai k.",
        "- Scatter PCA adalah proyeksi visual dua dimensi, bukan ruang yang dipakai untuk fitting.",
        "- Label angka cluster antaralgoritma tidak memiliki korespondensi langsung.",
        "- Kesimpulan akhir harus mempertimbangkan metrik, profil RFM, ukuran cluster, dan batasan data.",
    ]
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n" + "=" * 76)
    print("PERBANDINGAN SELESAI")
    print("=" * 76)
    print("\nK terbaik masing-masing algoritma:")
    print(best_comparison.round(4).to_string(index=False))
    print("\nProfil K-Means:")
    print(km_profile.to_string())
    print("\nProfil Hierarchical:")
    print(hc_profile.to_string())
    print("\nSemua perbandingan k:")
    print(comparison_all.round(4).to_string(index=False))
    print("\nLaporan:", REPORT_PATH)

if __name__ == "__main__":
    main()
