# 04_kmeans.py
# Eksperimen K-Means pada fitur RFM yang sudah ditransformasi dan distandardisasi.

from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "rfm_scaled.csv"
RFM_PATH = SCRIPT_DIR / "hasil_rfm.csv"
METRICS_PATH = SCRIPT_DIR / "kmeans_metrics.csv"
LABELS_PATH = SCRIPT_DIR / "hasil_kmeans.csv"
PROFILE_PATH = SCRIPT_DIR / "profil_cluster_kmeans.csv"
ELBOW_PATH = SCRIPT_DIR / "grafik_elbow_kmeans.png"
SILHOUETTE_PATH = SCRIPT_DIR / "grafik_silhouette_kmeans.png"
SCATTER_PATH = SCRIPT_DIR / "scatter_kmeans.png"

SCALED_COLUMNS = ["R_scaled", "F_scaled", "M_scaled"]
K_VALUES = range(2, 11)
RANDOM_STATE = 42

def main():
    if not DATA_PATH.is_file() or not RFM_PATH.is_file():
        raise FileNotFoundError("Jalankan 02_perhitungan_rfm.py dan 03_preprocessing.py terlebih dahulu.")

    scaled = pd.read_csv(DATA_PATH)
    original = pd.read_csv(RFM_PATH)
    if len(scaled) != len(original):
        raise ValueError("Jumlah baris rfm_scaled.csv dan hasil_rfm.csv berbeda.")
    X = scaled[SCALED_COLUMNS].to_numpy()
    if len(X) < 3:
        raise ValueError("Data pelanggan terlalu sedikit untuk clustering.")

    max_k = min(max(K_VALUES), len(X) - 1)
    tested_k = [k for k in K_VALUES if k <= max_k]
    if len(tested_k) < 1:
        raise ValueError("Tidak ada nilai k yang dapat diuji.")

    records = []
    fitted_models = {}
    print("Menjalankan eksperimen K-Means...")
    for k in tested_k:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        start = time.perf_counter()
        labels = model.fit_predict(X)
        elapsed = time.perf_counter() - start

        sil = silhouette_score(X, labels)
        dbi = davies_bouldin_score(X, labels)
        records.append({
            "k": k,
            "inertia": float(model.inertia_),
            "silhouette_score": float(sil),
            "davies_bouldin_index": float(dbi),
            "fit_time_seconds": float(elapsed),
        })
        fitted_models[k] = (model, labels)
        print(f"k={k:2d} | silhouette={sil:.4f} | DBI={dbi:.4f} | waktu={elapsed:.3f} detik")

    metrics = pd.DataFrame(records)
    metrics.to_csv(METRICS_PATH, index=False)

    # Pemilihan k dilakukan berdasarkan Silhouette Score tertinggi pada rentang uji.
    best_row = metrics.loc[metrics["silhouette_score"].idxmax()]
    best_k = int(best_row["k"])
    best_model, best_labels = fitted_models[best_k]

    result = original.copy()
    result["Cluster"] = best_labels
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

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(metrics["k"], metrics["inertia"], marker="o")
    ax.set_title("Elbow Method - K-Means")
    ax.set_xlabel("Jumlah cluster (k)")
    ax.set_ylabel("Inertia / WCSS")
    ax.set_xticks(metrics["k"])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ELBOW_PATH, dpi=160, bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(metrics["k"], metrics["silhouette_score"], marker="o")
    ax.axvline(best_k, linestyle="--", label=f"k terpilih = {best_k}")
    ax.set_title("Silhouette Score - K-Means")
    ax.set_xlabel("Jumlah cluster (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_xticks(metrics["k"])
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(SILHOUETTE_PATH, dpi=160, bbox_inches="tight")
    plt.show()

    # PCA hanya untuk memproyeksikan 3 fitur RFM menjadi 2 dimensi untuk visualisasi.
    # PCA tidak mengganti data yang dipakai model clustering.
    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=best_labels, cmap="tab10", s=14, alpha=0.7)
    ax.set_title(f"Visualisasi Cluster K-Means (PCA 2D), k={best_k}")
    ax.set_xlabel("Komponen Utama 1")
    ax.set_ylabel("Komponen Utama 2")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(SCATTER_PATH, dpi=180, bbox_inches="tight")
    plt.show()

    print("\n" + "=" * 72)
    print(f"HASIL K-MEANS | k terbaik berdasarkan silhouette = {best_k}")
    print("=" * 72)
    print(metrics.round(4).to_string(index=False))
    print("\nProfil cluster (nilai RFM asli):")
    print(profile.to_string())
    print(f"\nHasil label       : {LABELS_PATH}")
    print(f"Profil cluster    : {PROFILE_PATH}")
    print(f"Grafik Elbow      : {ELBOW_PATH}")
    print(f"Grafik Silhouette : {SILHOUETTE_PATH}")
    print(f"Scatter Plot      : {SCATTER_PATH}")

if __name__ == "__main__":
    main()
