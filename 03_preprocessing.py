# 03_preprocessing.py
# Preprocessing fitur RFM sebelum K-Means dan Hierarchical Clustering.
# Menggunakan log1p + StandardScaler; tidak membuang pelanggan outlier.

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "hasil_rfm.csv"
OUTPUT_PATH = SCRIPT_DIR / "rfm_scaled.csv"
INFO_PATH = SCRIPT_DIR / "preprocessing_info.json"
PLOT_PATH = SCRIPT_DIR / "visualisasi_preprocessing.png"

CUSTOMER_COLUMN = "Customer"
FEATURES = ["Recency", "Frequency", "Monetary"]

def main():
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"{INPUT_PATH} tidak ditemukan. Jalankan 02_perhitungan_rfm.py terlebih dahulu."
        )

    rfm = pd.read_csv(INPUT_PATH)
    required = [CUSTOMER_COLUMN] + FEATURES
    missing = [col for col in required if col not in rfm.columns]
    if missing:
        raise KeyError(f"Kolom berikut tidak ditemukan di hasil_rfm.csv: {missing}")

    original_rows = len(rfm)
    for col in FEATURES:
        rfm[col] = pd.to_numeric(rfm[col], errors="coerce")

    rfm = rfm.replace([np.inf, -np.inf], np.nan)
    rfm = rfm.dropna(subset=FEATURES + [CUSTOMER_COLUMN]).copy()
    removed_invalid = original_rows - len(rfm)

    # log1p hanya valid untuk nilai >= 0. Nilai negatif menandakan masalah data.
    negative_counts = {col: int((rfm[col] < 0).sum()) for col in FEATURES}
    if any(negative_counts.values()):
        raise ValueError(
            "Ditemukan nilai RFM negatif. Periksa hasil_rfm.csv: "
            f"{negative_counts}"
        )
    if (rfm[FEATURES] < 0).any().any():
        raise ValueError("Fitur RFM harus bernilai non-negatif sebelum log1p.")
    if len(rfm) < 3:
        raise ValueError("Jumlah pelanggan valid kurang dari 3; clustering tidak dapat dilakukan.")

    # Transformasi log1p mengurangi kemencengan/skewness tanpa menghapus pelanggan bernilai tinggi.
    log_values = np.log1p(rfm[FEATURES])
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(log_values)

    output = pd.DataFrame({
        CUSTOMER_COLUMN: rfm[CUSTOMER_COLUMN].values,
        "R_log": log_values["Recency"].values,
        "F_log": log_values["Frequency"].values,
        "M_log": log_values["Monetary"].values,
        "R_scaled": scaled_values[:, 0],
        "F_scaled": scaled_values[:, 1],
        "M_scaled": scaled_values[:, 2],
    })
    output.to_csv(OUTPUT_PATH, index=False)

    info = {
        "input_file": INPUT_PATH.name,
        "output_file": OUTPUT_PATH.name,
        "rows_before_cleaning": int(original_rows),
        "rows_after_cleaning": int(len(output)),
        "invalid_rows_removed": int(removed_invalid),
        "features": FEATURES,
        "transform": "natural logarithm log1p",
        "scaling": "StandardScaler (mean=0, standard deviation=1)",
        "outlier_policy": "Tidak menghapus outlier; log1p digunakan untuk mengurangi skewness.",
        "scaled_feature_columns": ["R_scaled", "F_scaled", "M_scaled"],
    }
    INFO_PATH.write_text(json.dumps(info, indent=4, ensure_ascii=False), encoding="utf-8")

    # Visualisasi distribusi sebelum dan sesudah log transform.
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for idx, feature in enumerate(FEATURES):
        axes[0, idx].hist(rfm[feature], bins=30, edgecolor="black")
        axes[0, idx].set_title(f"{feature} - Sebelum log")
        axes[0, idx].set_xlabel(feature)
        axes[0, idx].set_ylabel("Jumlah pelanggan")

        axes[1, idx].hist(log_values[feature], bins=30, edgecolor="black")
        axes[1, idx].set_title(f"{feature} - Setelah log1p")
        axes[1, idx].set_xlabel(f"log1p({feature})")
        axes[1, idx].set_ylabel("Jumlah pelanggan")

    fig.suptitle("Pemeriksaan Transformasi Fitur RFM", fontsize=15)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=160, bbox_inches="tight")
    plt.show()

    print("=" * 72)
    print("PREPROCESSING RFM SELESAI")
    print("=" * 72)
    print(f"Pelanggan sebelum preprocessing : {original_rows:,}")
    print(f"Pelanggan yang digunakan        : {len(output):,}")
    print(f"Baris tidak valid yang dibuang  : {removed_invalid:,}")
    print("\nStatistik fitur setelah standardisasi:")
    print(output[["R_scaled", "F_scaled", "M_scaled"]].describe().round(4))
    print(f"\nFile data siap clustering : {OUTPUT_PATH}")
    print(f"Ringkasan preprocessing   : {INFO_PATH}")
    print(f"Grafik preprocessing      : {PLOT_PATH}")

if __name__ == "__main__":
    main()
