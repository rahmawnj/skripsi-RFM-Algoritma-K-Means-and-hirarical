# 05_hierarchical.py
# Agglomerative Hierarchical Clustering dengan linkage Ward.
# Dendrogram dibuat dari sampel deterministik agar visualisasi tetap terbaca.

from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, dendrogram

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "rfm_scaled.csv"
RFM_PATH = SCRIPT_DIR / "hasil_rfm.csv"
KMEANS_METRICS_PATH = SCRIPT_DIR / "kmeans_metrics.csv"
METRICS_PATH = SCRIPT_DIR / "hierarchical_metrics.csv"
LABELS_PATH = SCRIPT_DIR / "hasil_hierarchical.csv"
PROFILE_PATH = SCRIPT_DIR / "profil_cluster_hierarchical.csv"
DENDROGRAM_PATH = SCRIPT_DIR / "dendrogram_hierarchical.png"
SCATTER_PATH = SCRIPT_DIR / "scatter_hierarchical.png"

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

    best_k = int(kmeans_metrics.loc[kmeans_metrics["silhouette_score"].idxmax(), "k"])
    X = scaled[SCALED_COLUMNS].to_numpy()
    n_samples = len(X)
    if not 2 <= best_k < n_samples:
        raise ValueError(f"Jumlah cluster k={best_k} tidak valid untuk {n_samples} pelanggan.")

    # Graph tetangga terdekat membatasi konektivitas saat penggabungan cluster.
    # Ini membantu mengurangi beban memori dibandingkan matriks jarak penuh.
    neighbors = min(CONNECTIVITY_NEIGHBORS, n_samples - 1)
    connectivity = kneighbors_graph(
        X, n_neighbors=neighbors, mode="connectivity", include_self=False
    )
    model = AgglomerativeClustering(
        n_clusters=best_k,
        metric="euclidean",
        linkage="ward",
        connectivity=connectivity,
        compute_full_tree=True,
    )

    print(f"Menjalankan Agglomerative Hierarchical Clustering untuk {n_samples:,} pelanggan...")
    start = time.perf_counter()
    labels = model.fit_predict(X)
    elapsed = time.perf_counter() - start

    sil = silhouette_score(X, labels)
    dbi = davies_bouldin_score(X, labels)
    metrics = pd.DataFrame([{
        "k": best_k,
        "linkage": "ward",
        "metric": "euclidean",
        "connectivity_neighbors": neighbors,
        "silhouette_score": float(sil),
        "davies_bouldin_index": float(dbi),
        "fit_time_seconds": float(elapsed),
    }])
    metrics.to_csv(METRICS_PATH, index=False)

    result = original.copy()
    result["Cluster"] = labels
    result.to_csv(LABELS_PATH, index=False)

    profile = result.groupby("Cluster").agg(
        Jumlah_Pelanggan=("Customer", "count"),
        Rata_Rata_Recency=("Recency", "mean"),
        Median_Recency=("Recency", "median"),
        Rata_Rata_Frequency=("Frequency", "mean"),
        Median_Frequency=("Frequency", "median"),
        Rata_Rata_Monetary=("Monetary", "mean"),
        Median_Monetary=("Monetary", "median"),
    ).round(2)
    profile.to_csv(PROFILE_PATH)

    # Dendrogram pada sampel acak tetap (seed 42), bukan seluruh pelanggan.
    # Model dan evaluasi di atas tetap dijalankan pada seluruh pelanggan.
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
    ax.set_title(f"Dendrogram Hierarchical Clustering (sampel {sample_size:,} pelanggan)")
    ax.set_xlabel("Kelompok / jumlah pelanggan pada cabang")
    ax.set_ylabel("Jarak penggabungan (Ward)")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(DENDROGRAM_PATH, dpi=180, bbox_inches="tight")
    plt.show()

    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=14, alpha=0.7)
    ax.set_title(f"Visualisasi Cluster Hierarchical (PCA 2D), k={best_k}")
    ax.set_xlabel("Komponen Utama 1")
    ax.set_ylabel("Komponen Utama 2")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(SCATTER_PATH, dpi=180, bbox_inches="tight")
    plt.show()

    print("\n" + "=" * 72)
    print("HASIL HIERARCHICAL CLUSTERING")
    print("=" * 72)
    print(f"Jumlah pelanggan      : {n_samples:,}")
    print(f"Jumlah cluster        : {best_k} (disamakan dengan K-Means)")
    print("Linkage               : Ward")
    print("Distance metric       : Euclidean")
    print(f"Connectivity neighbors: {neighbors}")
    print(f"Silhouette Score      : {sil:.4f}")
    print(f"Davies-Bouldin Index  : {dbi:.4f}")
    print(f"Waktu fit             : {elapsed:.3f} detik")
    print("\nProfil cluster (nilai RFM asli):")
    print(profile.to_string())
    print("\nCatatan: dendrogram divisualisasikan dari sampel pelanggan, sedangkan model dan metrik dihitung pada seluruh data.")
    print(f"\nHasil label    : {LABELS_PATH}")
    print(f"Profil cluster : {PROFILE_PATH}")
    print(f"Dendrogram     : {DENDROGRAM_PATH}")
    print(f"Scatter Plot   : {SCATTER_PATH}")

if __name__ == "__main__":
    main()
