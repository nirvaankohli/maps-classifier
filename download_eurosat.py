import os
import urllib.request
import zipfile

EUROSAT_URL = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"
DATA_DIR = "data/EuroSAT"
ZIP_PATH = "data/EuroSAT.zip"

os.makedirs("data", exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def download_file(url, dest):

    if os.path.exists(dest):

        print(f"File already exists: {dest}")
        return

    print(f"Downloading {url} to {dest}...")

    urllib.request.urlretrieve(url, dest)

    print("Download complete.")

def extract_zip(zip_path, extract_to):

    print(f"Extracting {zip_path} to {extract_to}...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:

        zip_ref.extractall(extract_to)

    print("Extraction complete.")

def main():

    download_file(EUROSAT_URL, ZIP_PATH)
    extract_zip(ZIP_PATH, "data/")

    print("EuroSat dataset is ready in 'data/EuroSAT'.")

if __name__ == "__main__":
    main() 