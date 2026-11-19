# 06_evaluasi_perbandingan.py
# Memilih jumlah cluster bersama berdasarkan rata-rata Silhouette kedua algoritma,
# lalu membandingkan metrik, profil, ukuran cluster, dan scatter plot.

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

COMPARISON_PATH = SCRIPT_DIR / "perbandingan_algoritma.csv"
K_SELECTION_PATH = SCRIPT_DIR / "pemilihan_k_bersama.csv"
KMEANS_LABELS_PATH = SCRIPT_DIR / "hasil_kmeans.csv"
HIERARCHICAL_LABELS_PATH = SCRIPT_DIR / "hasil_hierarchical.csv"
KMEANS_PROFILE_PATH = SCRIPT_DIR / "profil_cluster_kmeans.csv"
HIERARCHICAL_PROFILE_PATH = SCRIPT_DIR / "profil_cluster_hierarchical.csv"
COMPARISON_CHART_PATH = SCRIPT_DIR / "grafik_perbandingan_algoritma.png"
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
    points = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=14, alpha=0.7)
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
        raise ValueError("Jumlah baris antarfile tidak sama. Ulangi preprocessing dan clustering dari awal.")
    if not (
        original["Customer"].astype(str).equals(scaled["Customer"].astype(str))
        and original["Customer"].astype(str).equals(km_all["Customer"].astype(str))
        and original["Customer"].astype(str).equals(hc_all["Customer"].astype(str))
    ):
        raise ValueError("Urutan Customer antarfile berbeda. Jangan lanjutkan perbandingan sebelum data diselaraskan.")

    # Hanya nilai k yang diuji oleh kedua algoritma yang dapat dibandingkan.
    common = km_metrics[["k", "silhouette_score"]].merge(
        hc_metrics[["k", "silhouette_score"]], on="k", suffixes=("_kmeans", "_hierarchical")
    )
    if common.empty:
        raise ValueError("Tidak ada jumlah cluster (k) yang sama pada kedua hasil eksperimen.")

    common["mean_silhouette"] = (
        common["silhouette_score_kmeans"] + common["silhouette_score_hierarchical"]
    ) / 2
    common = common.sort_values("k")
    common.to_csv(K_SELECTION_PATH, index=False)
    selected_k = int(common.loc[common["mean_silhouette"].idxmax(), "k"])

    km_row = km_metrics.loc[km_metrics["k"] == selected_k].iloc[0]
    hc_row = hc_metrics.loc[hc_metrics["k"] == selected_k].iloc[0]
    km_labels = km_all[f"Cluster_k{selected_k}"].to_numpy()
    hc_labels = hc_all[f"Cluster_k{selected_k}"].to_numpy()

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

    comparison = pd.DataFrame([
        {
            "Algoritma": "K-Means",
            "Jumlah_Cluster": selected_k,
            "Silhouette_Score": float(km_row["silhouette_score"]),
            "Davies_Bouldin_Index": float(km_row["davies_bouldin_index"]),
            "Waktu_Fit_Detik": float(km_row["fit_time_seconds"]),
        },
        {
            "Algoritma": "Agglomerative Hierarchical",
            "Jumlah_Cluster": selected_k,
            "Silhouette_Score": float(hc_row["silhouette_score"]),
            "Davies_Bouldin_Index": float(hc_row["davies_bouldin_index"]),
            "Waktu_Fit_Detik": float(hc_row["fit_time_seconds"]),
        },
    ])
    comparison.to_csv(COMPARISON_PATH, index=False)

    # Perbandingan metrik pada jumlah cluster yang sama.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metric_specs = [
        ("Silhouette_Score", "Silhouette Score (lebih tinggi lebih baik)"),
        ("Davies_Bouldin_Index", "Davies-Bouldin Index (lebih rendah lebih baik)"),
        ("Waktu_Fit_Detik", "Waktu fit (detik, lebih rendah lebih cepat)"),
    ]
    for ax, (column, title) in zip(axes, metric_specs):
        bars = ax.bar(comparison["Algoritma"], comparison[column])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, comparison[column]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value:.4f}",
                    ha="center", va="bottom", fontsize=9)
    fig.suptitle(f"Perbandingan Algoritma pada k={selected_k}", fontsize=14)
    fig.tight_layout()
    fig.savefig(COMPARISON_CHART_PATH, dpi=180, bbox_inches="tight")
    plt.show()

    X = scaled[SCALED_COLUMNS].to_numpy()
    make_scatter(X, km_labels, f"Cluster K-Means (PCA 2D), k={selected_k}", KMEANS_SCATTER_PATH)
    make_scatter(X, hc_labels, f"Cluster Hierarchical (PCA 2D), k={selected_k}", HIERARCHICAL_SCATTER_PATH)

    # Perbandingan jumlah pelanggan per cluster.
    km_sizes = km_result["Cluster"].value_counts().sort_index()
    hc_sizes = hc_result["Cluster"].value_counts().sort_index()
    size_table = pd.DataFrame({
        "K-Means": km_sizes,
        "Hierarchical": hc_sizes,
    }).fillna(0)
    ax = size_table.plot(kind="bar", figsize=(9, 5))
    ax.set_title(f"Jumlah Pelanggan per Cluster (k={selected_k})")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Jumlah pelanggan")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(SIZE_CHART_PATH, dpi=180, bbox_inches="tight")
    plt.show()

    best_sil = comparison.loc[comparison["Silhouette_Score"].idxmax(), "Algoritma"]
    best_dbi = comparison.loc[comparison["Davies_Bouldin_Index"].idxmin(), "Algoritma"]
    fastest = comparison.loc[comparison["Waktu_Fit_Detik"].idxmin(), "Algoritma"]

    report_lines = [
        "RINGKASAN PERBANDINGAN CLUSTERING",
        "=" * 48,
        f"Jumlah pelanggan dianalisis: {len(original):,}",
        f"Jumlah cluster bersama terpilih: {selected_k}",
        "Dasar pemilihan k: rata-rata Silhouette Score K-Means dan Hierarchical pada k yang sama.",
        "",
        "Tabel perbandingan:",
        comparison.to_string(index=False),
        "",
        "Pemilihan k bersama:",
        common.to_string(index=False),
        "",
        f"Silhouette Score tertinggi: {best_sil}",
        f"Davies-Bouldin Index terendah: {best_dbi}",
        f"Waktu fit tercepat: {fastest}",
        "",
        "Interpretasi:",
        "- Silhouette lebih tinggi umumnya menunjukkan cluster lebih terpisah dan kompak.",
        "- Davies-Bouldin lebih rendah umumnya menunjukkan cluster lebih baik.",
        "- Waktu fit mengukur fitting algoritma saja, bukan preprocessing atau evaluasi.",
        "- Kesimpulan akhir tetap harus mempertimbangkan profil RFM, ukuran cluster, dan konteks penelitian.",
    ]
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"PERBANDINGAN SELESAI | k bersama = {selected_k}")
    print("=" * 72)
    print("\nTabel metrik:")
    print(comparison.round(4).to_string(index=False))
    print("\nProfil K-Means (RFM asli):")
    print(km_profile.to_string())
    print("\nProfil Hierarchical (RFM asli):")
    print(hc_profile.to_string())
    print("\n" + "\n".join(report_lines))
    print("\nFile utama:")
    for path in [
        K_SELECTION_PATH, COMPARISON_PATH, KMEANS_LABELS_PATH, HIERARCHICAL_LABELS_PATH,
        KMEANS_PROFILE_PATH, HIERARCHICAL_PROFILE_PATH, COMPARISON_CHART_PATH,
        KMEANS_SCATTER_PATH, HIERARCHICAL_SCATTER_PATH, SIZE_CHART_PATH, REPORT_PATH,
    ]:
        print("-", path.name)

if __name__ == "__main__":
    main()
