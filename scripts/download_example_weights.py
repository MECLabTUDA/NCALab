import requests  # type: ignore[import-untyped]
import zipfile
import hashlib
from pathlib import Path


def download_zip(url, download_path):
    response = requests.get(url)
    response.raise_for_status()
    with open(download_path, "wb") as zip_file:
        zip_file.write(response.content)


def extract_zip(zip_path, extract_to):
    Path(extract_to).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def verify_sha256(file_path, expected_sha256):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    calculated_sha256 = sha256_hash.hexdigest()

    if calculated_sha256 == expected_sha256:
        print("SHA256 checksum verified successfully.")
    else:
        print("SHA256 checksum verification failed.")
        raise ValueError("Checksum does not match!")


if __name__ == "__main__":
    zip_url = "https://github.com/MECLabTUDA/NCALab/releases/download/v0.3.2/pretrained_weights.zip"
    root_path = Path(__file__).parent / ".."
    download_path = (root_path / "pretrained_weights.zip").resolve()
    extract_to = root_path

    expected_sha256 = "7122e784d57341d2832ed8a34deb0c308a860d47242109e22a424d2572eb42f5"

    download_zip(zip_url, download_path)
    verify_sha256(download_path, expected_sha256)
    extract_zip(download_path, extract_to)
