#!/usr/bin/env python3
import zipfile
from pathlib import Path

import click


def zip_pth_files(zip_filename):
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for pth_file in Path("tasks").rglob("*.pth"):
            zipf.write(pth_file, pth_file.relative_to(Path(".")))


@click.command
@click.option(
    "--output", "-o", help="Output *.zip filename", default="pretrained_weights.zip"
)
def main(output):
    zip_filename = output
    zip_path = (Path(__file__) / ".." / ".." / zip_filename).resolve()
    zip_pth_files(zip_path)
    click.secho(f"Done. You'll find all pretrained weights in {zip_path}.", fg="green")
