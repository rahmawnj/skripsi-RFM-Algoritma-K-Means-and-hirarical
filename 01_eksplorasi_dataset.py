# 01_eksplorasi_dataset.py
# Jalankan dari Google Colab.
# Tujuan: mengunggah/membaca dataset dan memeriksa struktur data sebelum preprocessing.

from pathlib import Path
import pandas as pd

# Jika file sudah ada di folder Colab, isi nama/path file di sini.
DATA_PATH = ""  # Contoh: "/content/data_penjualan.csv"


def upload_dataset():
    """Memunculkan dialog upload jika dijalankan di Google Colab."""
    try:
        from google.colab import files
        uploaded = files.upload()
        if not uploaded:
            raise RuntimeError("Tidak ada file yang diunggah.")
        return f"/content/{next(iter(uploaded.keys()))}"
    except ImportError as exc:
        raise RuntimeError(
            "Upload otomatis hanya tersedia di Google Colab. "
            "Isi DATA_PATH dengan lokasi dataset."
        ) from exc


def load_dataset(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        # Mencoba UTF-8 terlebih dahulu, lalu latin-1 untuk file CSV tertentu.
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1")
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError("Format belum didukung. Gunakan CSV, XLSX, XLS, atau Parquet.")


def inspect_dataset(dataframe):
    print("=" * 70)
    print("RINGKASAN DATASET")
    print("=" * 70)
    print(f"Jumlah baris   : {dataframe.shape[0]:,}")
    print(f"Jumlah kolom   : {dataframe.shape[1]:,}")
    print(f"Duplikasi baris: {dataframe.duplicated().sum():,}")

    print("\nDAFTAR KOLOM")
    for index, column in enumerate(dataframe.columns, start=1):
        print(f"{index:>2}. {column!r} ({dataframe[column].dtype})")

    print("\nINFORMASI DATA")
    dataframe.info()

    print("\nMISSING VALUES")
    missing = dataframe.isna().sum().sort_values(ascending=False)
    print(missing[missing > 0] if missing.any() else "Tidak ada missing value.")

    print("\nSTATISTIK DESKRIPTIF")
    print(dataframe.describe(include="all").transpose())

    print("\n5 BARIS PERTAMA")
    display(dataframe.head())


if __name__ == "__main__":
    dataset_path = DATA_PATH.strip() or upload_dataset()
    df = load_dataset(dataset_path)
    inspect_dataset(df)
