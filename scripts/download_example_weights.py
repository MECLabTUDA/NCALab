import requests  # type: ignore[import-untyped]
import zipfile
import hashlib
from pathlib import Path
import tomllib


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
    with open("pyproject.toml", "rb") as f:
        release_version = tomllib.load(f)["project"]["version"]

    zip_url = f"https://github.com/MECLabTUDA/NCALab/releases/download/v{release_version}/pretrained_weights.zip"
    root_path = Path(__file__).parent / ".."
    download_path = (root_path / "pretrained_weights.zip").resolve()
    extract_to = root_path

    expected_sha256 = "c66fce525a97870babf94725d34bcd2a8619f42567180461755408d4b9fef3b5"

    download_zip(zip_url, download_path)
    verify_sha256(download_path, expected_sha256)
    extract_zip(download_path, extract_to)
