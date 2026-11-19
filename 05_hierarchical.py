# 05_hierarchical.py
# Agglomerative Hierarchical Clustering (Ward, Euclidean).
# Semua nilai k diuji agar perbandingan dengan K-Means netral.

from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, dendrogram

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "rfm_scaled.csv"
RFM_PATH = SCRIPT_DIR / "hasil_rfm.csv"
KMEANS_METRICS_PATH = SCRIPT_DIR / "kmeans_metrics.csv"
METRICS_PATH = SCRIPT_DIR / "hierarchical_metrics.csv"
ALL_LABELS_PATH = SCRIPT_DIR / "hasil_hierarchical_semua_k.csv"
DENDROGRAM_PATH = SCRIPT_DIR / "dendrogram_hierarchical.png"

SCALED_COLUMNS = ["R_scaled", "F_scaled", "M_scaled"]
RANDOM_STATE = 42
DENDROGRAM_SAMPLE_SIZE = 1000
CONNECTIVITY_NEIGHBORS = 10

def main():
    for path in [DATA_PATH, RFM_PATH, KMEANS_METRICS_PATH]:
        if not path.is_file():
            raise FileNotFoundError(f"{path.name} belum tersedia. Jalankan file sebelumnya.")

    scaled = pd.read_csv(DATA_PATH)
    original = pd.read_csv(RFM_PATH)
    kmeans_metrics = pd.read_csv(KMEANS_METRICS_PATH)
    if len(scaled) != len(original):
        raise ValueError("Jumlah baris data scaled dan RFM asli berbeda.")

    X = scaled[SCALED_COLUMNS].to_numpy()
    n_samples = len(X)
    tested_k = [int(k) for k in kmeans_metrics["k"].tolist() if int(k) < n_samples]
    if not tested_k:
        raise ValueError("Tidak ada nilai k valid dari hasil eksperimen K-Means.")

    # KNN connectivity membatasi tetangga yang boleh digabung pada tiap langkah.
    # Graph dibuat sekali dan digunakan kembali untuk setiap jumlah cluster.
    neighbors = min(CONNECTIVITY_NEIGHBORS, n_samples - 1)
    connectivity = kneighbors_graph(
        X, n_neighbors=neighbors, mode="connectivity", include_self=False
    )

    records = []
    all_labels = pd.DataFrame({"Customer": original["Customer"]})
    print(f"Menjalankan Hierarchical untuk {n_samples:,} pelanggan...")
    for k in tested_k:
        model = AgglomerativeClustering(
            n_clusters=k,
            metric="euclidean",
            linkage="ward",
            connectivity=connectivity,
            compute_full_tree=True,
        )
        start = time.perf_counter()
        labels = model.fit_predict(X)
        elapsed = time.perf_counter() - start

        silhouette_sample_size = min(5000, len(X))
        sil = silhouette_score(
            X, labels, sample_size=silhouette_sample_size, random_state=RANDOM_STATE
        )
        dbi = davies_bouldin_score(X, labels)
        records.append({
            "k": k,
            "linkage": "ward",
            "metric": "euclidean",
            "connectivity_neighbors": neighbors,
            "silhouette_score": float(sil),
            "silhouette_sample_size": silhouette_sample_size,
            "davies_bouldin_index": float(dbi),
            "fit_time_seconds": float(elapsed),
        })
        all_labels[f"Cluster_k{k}"] = labels
        print(f"k={k:2d} | silhouette={sil:.4f} | DBI={dbi:.4f} | waktu={elapsed:.3f} detik")

    metrics = pd.DataFrame(records)
    metrics.to_csv(METRICS_PATH, index=False)
    all_labels.to_csv(ALL_LABELS_PATH, index=False)

    # Dendrogram sampel deterministik; model dan metrik di atas tetap seluruh pelanggan.
    sample_size = min(DENDROGRAM_SAMPLE_SIZE, n_samples)
    rng = np.random.default_rng(RANDOM_STATE)
    sample_indices = np.sort(rng.choice(n_samples, size=sample_size, replace=False))
    sample_linkage = linkage(X[sample_indices], method="ward", metric="euclidean")

    fig, ax = plt.subplots(figsize=(14, 7))
    dendrogram(
        sample_linkage,
        truncate_mode="lastp",
        p=35,
        show_leaf_counts=True,
        leaf_rotation=45,
        leaf_font_size=9,
        ax=ax,
    )
    ax.set_title(f"Dendrogram Hierarchical (sampel {sample_size:,} pelanggan)")
    ax.set_xlabel("Kelompok / jumlah pelanggan pada cabang")
    ax.set_ylabel("Jarak penggabungan (Ward)")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(DENDROGRAM_PATH, dpi=180, bbox_inches="tight")
    plt.show()

    print("\n" + "=" * 72)
    print("HASIL UJI HIERARCHICAL CLUSTERING")
    print("=" * 72)
    print(metrics.round(4).to_string(index=False))
    print("\nLabel untuk semua nilai k disimpan di:", ALL_LABELS_PATH)
    print("K final belum dipilih. Tahap 06 memilih k bersama berdasarkan rata-rata Silhouette kedua algoritma.")
    print("Dendrogram sampel:", DENDROGRAM_PATH)

if __name__ == "__main__":
    main()
