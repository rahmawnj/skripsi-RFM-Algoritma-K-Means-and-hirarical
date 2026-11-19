# 01_eksplorasi_dataset.py
# Eksplorasi dataset Electronic Sales Sep 2023 - Sep 2024.
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

def inspect_dataset(df):
    print("\n" + "=" * 75)
    print("EKSPLORASI DATASET ELECTRONIC SALES")
    print("=" * 75)
    print(f"Jumlah baris          : {len(df):,}")
    print(f"Jumlah kolom          : {len(df.columns):,}")
    print(f"Jumlah baris duplikat : {df.duplicated().sum():,}")

    print("\nDAFTAR KOLOM DAN TIPE DATA")
    for number, column in enumerate(df.columns, start=1):
        print(f"{number:>2}. {column} ({df[column].dtype})")

    print("\nMISSING VALUES")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(missing.to_string() if not missing.empty else "Tidak ada missing value.")

    print("\nDISTRIBUSI ORDER STATUS")
    if "Order Status" in df.columns:
        print(df["Order Status"].value_counts(dropna=False).to_string())

    print("\nSTATISTIK DESKRIPTIF")
    print(df.describe(include="all").transpose().to_string())

    print("\n5 BARIS PERTAMA")
    try:
        from IPython.display import display
        display(df.head())
    except ImportError:
        print(df.head().to_string(index=False))

if __name__ == "__main__":
    dataset_path = find_dataset()
    data = read_dataset(dataset_path)
    inspect_dataset(data)
