# 02_perhitungan_rfm.py
# Jalankan setelah nama kolom dataset diketahui dari 01_eksplorasi_dataset.py.
# Tujuan: membentuk tabel Recency, Frequency, Monetary per pelanggan.

from pathlib import Path
import pandas as pd

# ========================= KONFIGURASI =========================
DATA_PATH = ""  # Contoh: "/content/data_penjualan.csv"

# Ganti nilai berikut sesuai nama kolom yang benar-benar ada di dataset.
CUSTOMER_COLUMN = "Customer ID"
TRANSACTION_COLUMN = "Transaction ID"
DATE_COLUMN = "Transaction Date"
AMOUNT_COLUMN = "Total Amount"

# Tanggal acuan Recency. None berarti satu hari setelah tanggal transaksi terakhir.
ANALYSIS_DATE = None  # Contoh: "2024-10-01"
OUTPUT_PATH = "/content/hasil_rfm.csv"
# ===============================================================


def read_data(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {path}. Isi DATA_PATH terlebih dahulu."
        )
    if path.suffix.lower() == ".csv":
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1")
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError("Gunakan dataset CSV atau Excel.")


def build_rfm(dataframe):
    required = [CUSTOMER_COLUMN, TRANSACTION_COLUMN, DATE_COLUMN, AMOUNT_COLUMN]
    missing_columns = [column for column in required if column not in dataframe.columns]
    if missing_columns:
        raise KeyError(
            "Kolom berikut tidak ditemukan: "
            + ", ".join(missing_columns)
            + "\nKolom tersedia: "
            + ", ".join(map(str, dataframe.columns))
            + "\nSesuaikan konfigurasi kolom di bagian atas file."
        )

    data = dataframe[required].copy()
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
    data[AMOUNT_COLUMN] = pd.to_numeric(data[AMOUNT_COLUMN], errors="coerce")

    # Baris yang tidak memiliki identitas pelanggan, transaksi, tanggal,
    # atau nominal valid tidak digunakan dalam perhitungan RFM.
    data = data.dropna(subset=required)
    data = data[data[AMOUNT_COLUMN] >= 0]

    if data.empty:
        raise ValueError("Tidak ada transaksi valid setelah pembersihan dasar.")

    if ANALYSIS_DATE:
        analysis_date = pd.Timestamp(ANALYSIS_DATE)
    else:
        analysis_date = data[DATE_COLUMN].max().normalize() + pd.Timedelta(days=1)

    rfm = (
        data.groupby(CUSTOMER_COLUMN)
        .agg(
            Recency=(DATE_COLUMN, lambda dates: (analysis_date - dates.max().normalize()).days),
            Frequency=(TRANSACTION_COLUMN, "nunique"),
            Monetary=(AMOUNT_COLUMN, "sum"),
        )
        .reset_index()
        .rename(columns={CUSTOMER_COLUMN: "Customer"})
    )

    rfm["Recency"] = rfm["Recency"].astype(int)
    rfm["Frequency"] = rfm["Frequency"].astype(int)
    rfm["Monetary"] = rfm["Monetary"].round(2)

    return rfm, analysis_date


if __name__ == "__main__":
    if not DATA_PATH.strip():
        raise ValueError("Isi DATA_PATH dengan lokasi dataset sebelum menjalankan file.")

    raw_data = read_data(DATA_PATH)
    rfm_data, reference_date = build_rfm(raw_data)

    print(f"Tanggal acuan Recency: {reference_date.date()}")
    print(f"Jumlah pelanggan     : {len(rfm_data):,}")
    print("\nRingkasan RFM:")
    display(rfm_data.head())

    print("\nStatistik RFM:")
    display(rfm_data[["Recency", "Frequency", "Monetary"]].describe())

    rfm_data.to_csv(OUTPUT_PATH, index=False)
    print(f"\nHasil RFM disimpan ke: {OUTPUT_PATH}")
