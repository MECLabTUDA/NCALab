#!/usr/bin/env python3
import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)
import hashlib
import shutil

import click
import requests
import tqdm

from pathlib import Path

KVASIR_SEG_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"
KVASIR_SEG_CHECKSUM = "03b30e21d584e04facf49397a2576738fd626815771afbbf788f74a7153478f7"
RANDOM_STATE = 1337

from ncalab.paths import ROOT_PATH

KVASIR_SEG_PATH = ROOT_PATH / "data" / "kvasir_seg"


def validate_checksum(filename: str):
    """_summary_

    Args:
        filename (str): _description_

    Returns:
        _type_: _description_
    """
    checksum = KVASIR_SEG_CHECKSUM
    if checksum is None:
        click.secho("No checksum available to compare with.", fg="yellow")
        click.secho("The downloaded file might be unsafe.", fg="yellow")
        if not click.confirm("Still want to continue?"):
            return False
        return True
    else:
        click.secho("Validating checksum...", fg="blue")
        sha256 = hashlib.sha256()
        with open(filename, "rb") as f:
            while True:
                data = f.read(65536)
                if not data:
                    break
                sha256.update(data)
        file_hash = sha256.hexdigest()
        if checksum != file_hash:
            click.secho(
                "Invalid checksum. This might be a bug, a server error or transmission problem.",
                fg="red",
            )
            click.secho(f"expected: {checksum}")
            click.secho(f"download: {file_hash}")
            return False
        click.secho("Checksum ok.", fg="green")
    return True


def download_kvasir_seg(filename: str):
    """_summary_

    Args:
        filename (str): _description_
    """
    url = KVASIR_SEG_URL
    destination = filename
    click.secho(f"File: {filename}", fg="blue")
    click.secho(f"Downloading file from URL {url}...", fg="blue")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in tqdm.tqdm(
                r.iter_content(chunk_size=1024),
                total=int(r.headers.get("content-length", 0)) // 1024,
                unit="KB",
            ):
                f.write(chunk)
    click.secho("Done.", fg="green")
    if not validate_checksum(filename):
        exit()
    click.secho("")


def extract_archive(filename, destination=None):
    """_summary_

    Args:
        filename (_type_): _description_
        destination (_type_, optional): _description_. Defaults to None.
    """
    click.secho(f"Extracting archive {filename}...", fg="blue")
    archive_path = filename
    if not destination:
        destination = filename[: -len(".zip")]
    shutil.unpack_archive(archive_path, destination)
    click.secho("Done.", fg="green")


def download_and_extract():
    """_summary_"""
    KVASIR_SEG_PATH.mkdir(exist_ok=True)
    if os.path.exists("kvasir_seg.zip"):
        if not validate_checksum("kvasir_seg.zip"):
            download_kvasir_seg("kvasir_seg.zip")
            validate_checksum("kvasir_seg.zip")
    else:
        download_kvasir_seg("kvasir_seg.zip")
        validate_checksum("kvasir_seg.zip")
    extract_archive("kvasir_seg.zip", KVASIR_SEG_PATH)
