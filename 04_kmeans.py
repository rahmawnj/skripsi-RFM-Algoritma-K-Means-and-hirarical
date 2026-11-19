# 04_kmeans.py
# Uji K-Means pada rentang k yang sama untuk dibandingkan dengan Hierarchical.

from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "rfm_scaled.csv"
RFM_PATH = SCRIPT_DIR / "hasil_rfm.csv"
METRICS_PATH = SCRIPT_DIR / "kmeans_metrics.csv"
ALL_LABELS_PATH = SCRIPT_DIR / "hasil_kmeans_semua_k.csv"
ELBOW_PATH = SCRIPT_DIR / "grafik_elbow_kmeans.png"
SILHOUETTE_PATH = SCRIPT_DIR / "grafik_silhouette_kmeans.png"
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

    tested_k = [k for k in K_VALUES if k < len(X)]
    records = []
    all_labels = pd.DataFrame({"Customer": original["Customer"]})
    silhouette_sample_size = min(5000, len(X))
    rng = np.random.default_rng(RANDOM_STATE)
    sample_indices = np.sort(rng.choice(len(X), size=silhouette_sample_size, replace=False))

    print("Menjalankan K-Means untuk setiap nilai k...")
    for k in tested_k:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        start = time.perf_counter()
        labels = model.fit_predict(X)
        elapsed = time.perf_counter() - start

        sil = silhouette_score(X[sample_indices], labels[sample_indices])
        dbi = davies_bouldin_score(X, labels)
        records.append({
            "k": k,
            "inertia": float(model.inertia_),
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

    # Grafik ini membantu membaca tren; k final dipilih setelah kedua algoritma diuji.
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
    ax.set_title("Silhouette Score K-Means untuk Setiap k")
    ax.set_xlabel("Jumlah cluster (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_xticks(metrics["k"])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(SILHOUETTE_PATH, dpi=160, bbox_inches="tight")
    plt.show()

    print("\n" + "=" * 72)
    print("HASIL UJI K-MEANS")
    print("=" * 72)
    print(metrics.round(4).to_string(index=False))
    print("\nLabel untuk semua nilai k disimpan di:", ALL_LABELS_PATH)
    print("K final belum dipilih di tahap ini. K final ditentukan setelah hasil Hierarchical tersedia.")
    print("Grafik Elbow:", ELBOW_PATH)
    print("Grafik Silhouette:", SILHOUETTE_PATH)

if __name__ == "__main__":
    main()
