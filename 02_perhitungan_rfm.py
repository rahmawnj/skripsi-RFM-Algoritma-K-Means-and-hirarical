# 02_perhitungan_rfm.py
# Menghitung RFM dari dataset Electronic Sales Sep 2023 - Sep 2024.
# Dataset dicari otomatis; tidak ada dialog upload.

from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILENAME = "Electronic_sales_Sep2023-Sep2024.csv"
KAGGLEHUB_PATH = (
    Path.home()
    / ".cache/kagglehub/datasets/cameronseamons/"
      "electronic-sales-sep2023-sep2024/versions/5"
    / DATA_FILENAME
)
OUTPUT_PATH = SCRIPT_DIR / "hasil_rfm.csv"

CUSTOMER_COLUMN = "Customer ID"
DATE_COLUMN = "Purchase Date"
AMOUNT_COLUMN = "Total Price"
STATUS_COLUMN = "Order Status"
ANALYSIS_DATE = None  # None = satu hari setelah tanggal pembelian terakhir.

def find_dataset():
    candidates = [
        SCRIPT_DIR / DATA_FILENAME,
        Path("/content") / DATA_FILENAME,
        KAGGLEHUB_PATH,
    ]
    for candidate in candidates:
        if candidate.is_file():
            print(f"Dataset ditemukan: {candidate}")
            return candidate

    searched = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        "Dataset CSV tidak ditemukan. Lokasi yang sudah diperiksa:\n"
        f"{searched}\n"
        "Pastikan file CSV berada satu folder dengan file Python."
    )

def read_dataset(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")

def calculate_rfm(raw):
    required = [CUSTOMER_COLUMN, DATE_COLUMN, AMOUNT_COLUMN]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise KeyError(
            f"Kolom wajib tidak ditemukan: {missing}\n"
            f"Kolom yang tersedia: {raw.columns.tolist()}"
        )

    data = raw.copy()
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
    data[AMOUNT_COLUMN] = pd.to_numeric(data[AMOUNT_COLUMN], errors="coerce")
    data = data.dropna(subset=required)

    # Order yang dibatalkan tidak dihitung sebagai pembelian.
    if STATUS_COLUMN in data.columns:
        status = data[STATUS_COLUMN].astype(str).str.strip().str.lower()
        before = len(data)
        data = data[status == "completed"].copy()
        print(f"Baris non-Completed yang tidak dihitung: {before - len(data):,}")

    data = data[data[AMOUNT_COLUMN] >= 0].copy()
    if data.empty:
        raise ValueError(
            "Tidak ada data pembelian valid. Periksa kolom tanggal, nominal, "
            "dan nilai Order Status."
        )

    data["_purchase_day"] = data[DATE_COLUMN].dt.normalize()
    if ANALYSIS_DATE is not None:
        analysis_date = pd.Timestamp(ANALYSIS_DATE).normalize()
    else:
        analysis_date = data["_purchase_day"].max() + pd.Timedelta(days=1)

    # Dataset ini tidak memiliki Transaction ID/order ID.
    # Frequency memakai jumlah tanggal pembelian unik per pelanggan.
    # Ini mencegah beberapa baris pada tanggal yang sama dihitung berkali-kali.
    rfm = (
        data.groupby(CUSTOMER_COLUMN)
        .agg(
            Recency=("_purchase_day", lambda dates: (analysis_date - dates.max()).days),
            Frequency=("_purchase_day", "nunique"),
            Monetary=(AMOUNT_COLUMN, "sum"),
        )
        .reset_index()
        .rename(columns={CUSTOMER_COLUMN: "Customer"})
    )

    rfm["Recency"] = rfm["Recency"].astype(int)
    rfm["Frequency"] = rfm["Frequency"].astype(int)
    rfm["Monetary"] = rfm["Monetary"].round(2)
    return rfm, analysis_date, len(data)

if __name__ == "__main__":
    dataset_path = find_dataset()
    raw_data = read_dataset(dataset_path)
    print(f"Jumlah baris dataset mentah: {len(raw_data):,}")

    rfm, reference_date, used_rows = calculate_rfm(raw_data)

    print("\n" + "=" * 65)
    print("HASIL PERHITUNGAN RFM")
    print("=" * 65)
    print(f"Tanggal acuan Recency  : {reference_date.date()}")
    print(f"Baris pembelian dipakai: {used_rows:,}")
    print(f"Jumlah pelanggan       : {len(rfm):,}")

    print("\n5 BARIS PERTAMA RFM")
    try:
        from IPython.display import display
        display(rfm.head())
        display(rfm[["Recency", "Frequency", "Monetary"]].describe())
    except ImportError:
        print(rfm.head().to_string(index=False))
        print(rfm[["Recency", "Frequency", "Monetary"]].describe().to_string())

    rfm.to_csv(OUTPUT_PATH, index=False)
    print(f"\nFile hasil RFM disimpan di: {OUTPUT_PATH}")
