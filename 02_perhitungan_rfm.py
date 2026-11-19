# 02_perhitungan_rfm.py
# Pembentukan fitur RFM untuk dataset Electronic Sales Sep 2023-Sep 2024.
# Dataset dicari otomatis di folder script, /content, atau cache KaggleHub.

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

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
ADDON_AMOUNT_COLUMN = "Add-on Total"
STATUS_COLUMN = "Order Status"
ANALYSIS_DATE = None

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
        "Pastikan CSV berada satu folder dengan file Python atau cache KaggleHub."
    )

def read_dataset(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")

def calculate_rfm(raw):
    required = [CUSTOMER_COLUMN, DATE_COLUMN, AMOUNT_COLUMN, STATUS_COLUMN]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise KeyError(
            f"Kolom wajib tidak ditemukan: {missing}\n"
            f"Kolom tersedia: {raw.columns.tolist()}"
        )

    data = raw.copy()
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
    data[AMOUNT_COLUMN] = pd.to_numeric(data[AMOUNT_COLUMN], errors="coerce")
    if ADDON_AMOUNT_COLUMN in data.columns:
        data[ADDON_AMOUNT_COLUMN] = pd.to_numeric(
            data[ADDON_AMOUNT_COLUMN], errors="coerce"
        ).fillna(0)
    else:
        data[ADDON_AMOUNT_COLUMN] = 0

    data = data.dropna(subset=[CUSTOMER_COLUMN, DATE_COLUMN, AMOUNT_COLUMN])
    status = data[STATUS_COLUMN].astype(str).str.strip().str.lower()
    before_status_filter = len(data)
    data = data[status == "completed"].copy()
    excluded_status = before_status_filter - len(data)

    data = data[
        (data[AMOUNT_COLUMN] >= 0)
        & (data[ADDON_AMOUNT_COLUMN] >= 0)
    ].copy()
    if data.empty:
        raise ValueError(
            "Tidak ada pesanan Completed yang valid. Periksa Order Status, "
            "Purchase Date, dan nilai harga."
        )

    # Monetary mencakup harga produk dan add-on.
    data["_monetary"] = data[AMOUNT_COLUMN] + data[ADDON_AMOUNT_COLUMN]

    if ANALYSIS_DATE is not None:
        analysis_date = pd.Timestamp(ANALYSIS_DATE).normalize()
    else:
        analysis_date = data[DATE_COLUMN].max().normalize() + pd.Timedelta(days=1)

    # Dataset tidak menyediakan Order ID/Transaction ID.
    # Dengan asumsi satu baris = satu pesanan, Frequency adalah jumlah
    # baris pesanan Completed per pelanggan, bukan jumlah tanggal unik.
    rfm = (
        data.groupby(CUSTOMER_COLUMN)
        .agg(
            Recency=(
                DATE_COLUMN,
                lambda dates: (analysis_date - dates.max().normalize()).days
            ),
            Frequency=(CUSTOMER_COLUMN, "size"),
            Monetary=("_monetary", "sum"),
        )
        .reset_index()
        .rename(columns={CUSTOMER_COLUMN: "Customer"})
    )

    rfm["Recency"] = rfm["Recency"].astype(int)
    rfm["Frequency"] = rfm["Frequency"].astype(int)
    rfm["Monetary"] = rfm["Monetary"].round(2)
    if (rfm["Recency"] < 0).any():
        raise ValueError("Ditemukan Recency negatif. Periksa ANALYSIS_DATE.")

    return rfm, analysis_date, len(data), excluded_status

if __name__ == "__main__":
    dataset_path = find_dataset()
    raw_data = read_dataset(dataset_path)
    print(f"Jumlah baris dataset mentah: {len(raw_data):,}")

    rfm, reference_date, used_rows, excluded_status = calculate_rfm(raw_data)

    print("\n" + "=" * 68)
    print("HASIL PEMBENTUKAN FITUR RFM")
    print("=" * 68)
    print(f"Tanggal acuan Recency        : {reference_date.date()}")
    print(f"Pesanan Completed digunakan : {used_rows:,}")
    print(f"Baris status selain Completed: {excluded_status:,}")
    print(f"Jumlah pelanggan unik        : {len(rfm):,}")

    print("\n5 BARIS PERTAMA")
    try:
        from IPython.display import display
        display(rfm.head())
        print("\nStatistik deskriptif RFM")
        display(rfm[["Recency", "Frequency", "Monetary"]].describe())
    except ImportError:
        print(rfm.head().to_string(index=False))
        print(rfm[["Recency", "Frequency", "Monetary"]].describe().to_string())

    rfm.to_csv(OUTPUT_PATH, index=False)
    print(f"\nHasil RFM disimpan di: {OUTPUT_PATH}")

    # Visualisasi awal distribusi RFM untuk pemeriksaan sebelum clustering.
    features = ["Recency", "Frequency", "Monetary"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, feature in zip(axes, features):
        ax.hist(rfm[feature], bins=30, edgecolor="black")
        ax.set_title(f"Distribusi {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Jumlah pelanggan")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Distribusi Fitur RFM Pelanggan", fontsize=14)
    fig.tight_layout()
    plot_path = SCRIPT_DIR / "visualisasi_distribusi_rfm.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Visualisasi disimpan di: {plot_path}")

