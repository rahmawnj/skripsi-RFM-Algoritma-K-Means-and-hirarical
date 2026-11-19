# 06_evaluasi_perbandingan.py
# Membandingkan K-Means dan Hierarchical Clustering pada jumlah cluster yang sama.

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
KMEANS_PATH = SCRIPT_DIR / "kmeans_metrics.csv"
HIERARCHICAL_PATH = SCRIPT_DIR / "hierarchical_metrics.csv"
KMEANS_LABELS_PATH = SCRIPT_DIR / "hasil_kmeans.csv"
HIERARCHICAL_LABELS_PATH = SCRIPT_DIR / "hasil_hierarchical.csv"
OUTPUT_PATH = SCRIPT_DIR / "perbandingan_algoritma.csv"
CHART_PATH = SCRIPT_DIR / "grafik_perbandingan_algoritma.png"
REPORT_PATH = SCRIPT_DIR / "ringkasan_perbandingan.txt"

def main():
    required = [KMEANS_PATH, HIERARCHICAL_PATH, KMEANS_LABELS_PATH, HIERARCHICAL_LABELS_PATH]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"{path.name} belum tersedia. Jalankan 04_kmeans.py dan 05_hierarchical.py.")

    km = pd.read_csv(KMEANS_PATH)
    hc = pd.read_csv(HIERARCHICAL_PATH)
    km_row = km.loc[km["silhouette_score"].idxmax()]
    hc_row = hc.iloc[0]

    if int(km_row["k"]) != int(hc_row["k"]):
        raise ValueError(
            "Jumlah cluster berbeda. Perbandingan harus menggunakan k yang sama: "
            f"K-Means={int(km_row['k'])}, Hierarchical={int(hc_row['k'])}."
        )

    comparison = pd.DataFrame([
        {
            "Algoritma": "K-Means",
            "Jumlah_Cluster": int(km_row["k"]),
            "Silhouette_Score": float(km_row["silhouette_score"]),
            "Davies_Bouldin_Index": float(km_row["davies_bouldin_index"]),
            "Waktu_Fit_Detik": float(km_row["fit_time_seconds"]),
        },
        {
            "Algoritma": "Agglomerative Hierarchical",
            "Jumlah_Cluster": int(hc_row["k"]),
            "Silhouette_Score": float(hc_row["silhouette_score"]),
            "Davies_Bouldin_Index": float(hc_row["davies_bouldin_index"]),
            "Waktu_Fit_Detik": float(hc_row["fit_time_seconds"]),
        },
    ])
    comparison.to_csv(OUTPUT_PATH, index=False)

    # Grafik perbandingan metrik; waktu fit dipisahkan karena skala satuannya berbeda.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [
        ("Silhouette_Score", "Silhouette Score (lebih tinggi lebih baik)"),
        ("Davies_Bouldin_Index", "Davies-Bouldin Index (lebih rendah lebih baik)"),
        ("Waktu_Fit_Detik", "Waktu fit (detik, lebih rendah lebih cepat)"),
    ]
    for ax, (column, title) in zip(axes, metrics):
        bars = ax.bar(comparison["Algoritma"], comparison[column])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, comparison[column]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value:.4f}",
                    ha="center", va="bottom", fontsize=9)
    fig.suptitle(f"Perbandingan Algoritma pada k={int(km_row['k'])}", fontsize=14)
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=180, bbox_inches="tight")
    plt.show()

    best_sil = comparison.loc[comparison["Silhouette_Score"].idxmax(), "Algoritma"]
    best_dbi = comparison.loc[comparison["Davies_Bouldin_Index"].idxmin(), "Algoritma"]
    fastest = comparison.loc[comparison["Waktu_Fit_Detik"].idxmin(), "Algoritma"]

    lines = [
        "RINGKASAN PERBANDINGAN CLUSTERING",
        "=" * 45,
        f"Jumlah cluster yang dibandingkan: {int(km_row['k'])}",
        "",
        comparison.to_string(index=False),
        "",
        f"Silhouette Score tertinggi : {best_sil}",
        f"Davies-Bouldin Index terendah: {best_dbi}",
        f"Waktu fit tercepat        : {fastest}",
        "",
        "Catatan interpretasi:",
        "- Silhouette lebih tinggi umumnya menunjukkan cluster lebih terpisah dan kompak.",
        "- Davies-Bouldin lebih rendah umumnya menunjukkan cluster lebih baik.",
        "- Waktu fit mengukur proses pelatihan/fitting, bukan seluruh waktu preprocessing atau evaluasi.",
        "- Kesimpulan akhir perlu mempertimbangkan profil RFM, ukuran cluster, dan keterbatasan dataset; jangan memilih algoritma hanya dari satu metrik.",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nTabel perbandingan : {OUTPUT_PATH}")
    print(f"Grafik perbandingan: {CHART_PATH}")
    print(f"Ringkasan          : {REPORT_PATH}")

if __name__ == "__main__":
    main()
