# 05_hierarchical.py
# Standard Agglomerative Hierarchical Clustering dengan Ward linkage dan Euclidean distance.
# Satu pohon hierarki dibangun sekali, lalu dipotong pada setiap nilai k.

from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
from sklearn.metrics import silhouette_score, davies_bouldin_score

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "rfm_scaled.csv"
RFM_PATH = SCRIPT_DIR / "hasil_rfm.csv"
KMEANS_METRICS_PATH = SCRIPT_DIR / "kmeans_metrics.csv"
METRICS_PATH = SCRIPT_DIR / "hierarchical_metrics.csv"
ALL_LABELS_PATH = SCRIPT_DIR / "hasil_hierarchical_semua_k.csv"
DENDROGRAM_PATH = SCRIPT_DIR / "dendrogram_hierarchical.png"

SCALED_COLUMNS = ["R_scaled", "F_scaled", "M_scaled"]
RANDOM_STATE = 42
K_VALUES = range(2, 11)
MAX_CUSTOMERS_FOR_FULL_HIERARCHY = 12000

def main():
    for path in [DATA_PATH, RFM_PATH, KMEANS_METRICS_PATH]:
        if not path.is_file():
            raise FileNotFoundError(f"{path.name} belum tersedia. Jalankan tahap sebelumnya.")

    scaled = pd.read_csv(DATA_PATH)
    original = pd.read_csv(RFM_PATH)
    kmeans_metrics = pd.read_csv(KMEANS_METRICS_PATH)

    if len(scaled) != len(original):
        raise ValueError("Jumlah baris data scaled dan RFM asli berbeda.")
    if not original["Customer"].astype(str).equals(scaled["Customer"].astype(str)):
        raise ValueError("Urutan Customer pada hasil_rfm.csv dan rfm_scaled.csv berbeda.")

    X = scaled[SCALED_COLUMNS].to_numpy(dtype=float)
    n_samples = len(X)
    if n_samples < 3:
        raise ValueError("Minimal diperlukan 3 pelanggan untuk clustering.")
    if n_samples > MAX_CUSTOMERS_FOR_FULL_HIERARCHY:
        raise MemoryError(
            f"Jumlah pelanggan {n_samples:,} melebihi batas aman skrip "
            f"({MAX_CUSTOMERS_FOR_FULL_HIERARCHY:,}) untuk full Ward linkage. "
            "Hierarchical standar membutuhkan memori kuadratik. Jangan mengganti "
            "diam-diam ke metode constrained; diskusikan strategi sampling/komputasi "
            "dengan pembimbing agar metodologi penelitian tetap jelas."
        )

    tested_k = [int(k) for k in K_VALUES if int(k) < n_samples]
    tested_k = [k for k in tested_k if k in set(kmeans_metrics["k"].astype(int))]
    if not tested_k:
        raise ValueError("Tidak ada nilai k bersama yang valid dengan eksperimen K-Means.")

    print(f"Membangun pohon Hierarchical Ward untuk {n_samples:,} pelanggan...")
    start = time.perf_counter()
    # Ward hanya valid dengan Euclidean distance.
    hierarchy = linkage(X, method="ward", metric="euclidean", optimal_ordering=False)
    tree_fit_seconds = time.perf_counter() - start
    print(f"Pohon hierarki selesai dalam {tree_fit_seconds:.3f} detik.")

    # Gunakan subset indeks yang sama untuk menghitung Silhouette semua algoritma.
    silhouette_sample_size = min(5000, n_samples)
    rng = np.random.default_rng(RANDOM_STATE)
    sample_indices = np.sort(
        rng.choice(n_samples, size=silhouette_sample_size, replace=False)
    )

    records = []
    all_labels = pd.DataFrame({"Customer": original["Customer"]})
    print("\nEvaluasi pemotongan pohon pada setiap jumlah cluster...")
    for k in tested_k:
        labels = cut_tree(hierarchy, n_clusters=[k]).reshape(-1).astype(int)
        sample_labels = labels[sample_indices]
        if len(np.unique(sample_labels)) < 2:
            sil = float("nan")
        else:
            sil = silhouette_score(X[sample_indices], sample_labels)

        dbi = davies_bouldin_score(X, labels)
        records.append({
            "k": k,
            "linkage": "ward",
            "metric": "euclidean",
            "silhouette_score": float(sil),
            "silhouette_sample_size": silhouette_sample_size,
            "davies_bouldin_index": float(dbi),
            "tree_fit_time_seconds": float(tree_fit_seconds),
            "label_cut_time_seconds": None,
        })
        all_labels[f"Cluster_k{k}"] = labels
        print(f"k={k:2d} | silhouette={sil:.4f} | DBI={dbi:.4f}")

    metrics = pd.DataFrame(records)
    metrics.to_csv(METRICS_PATH, index=False)
    all_labels.to_csv(ALL_LABELS_PATH, index=False)

    fig, ax = plt.subplots(figsize=(15, 8))
    dendrogram(
        hierarchy,
        truncate_mode="lastp",
        p=40,
        show_leaf_counts=True,
        leaf_rotation=45,
        leaf_font_size=9,
        ax=ax,
    )
    ax.set_title("Dendrogram Agglomerative Hierarchical - Ward")
    ax.set_xlabel("Kelompok / jumlah pelanggan pada cabang")
    ax.set_ylabel("Jarak penggabungan (Ward)")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(DENDROGRAM_PATH, dpi=180, bbox_inches="tight")
    plt.show()

    print("\n" + "=" * 72)
    print("HASIL HIERARCHICAL CLUSTERING")
    print("=" * 72)
    print(metrics.round(4).to_string(index=False))
    print(f"\nJumlah pelanggan: {n_samples:,}")
    print(f"Waktu membangun pohon Ward: {tree_fit_seconds:.4f} detik")
    print("Label semua k:", ALL_LABELS_PATH)
    print("Dendrogram dari pohon yang sama:", DENDROGRAM_PATH)

if __name__ == "__main__":
    main()
