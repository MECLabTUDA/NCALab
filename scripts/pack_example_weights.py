#!/usr/bin/env python3
import zipfile
from pathlib import Path

import click


def zip_pth_files(zip_filename):
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for pth_file in Path("tasks").rglob("*.pth"):
            click.secho(
                f"Packing {pth_file}... ({pth_file.stat().st_size / 1024:.2f} KiB)",
                fg="blue",
            )
            zipf.write(pth_file, pth_file.relative_to(Path(".")))


@click.command(
    help="Pack all weights (*.pth files) in subfolders of tasks/ into a single zip file."
)
@click.option(
    "--output", "-o", help="Output *.zip filename", default="pretrained_weights.zip"
)
def main(output):
    zip_filename = output
    zip_path = (Path(__file__) / ".." / ".." / zip_filename).resolve()
    zip_pth_files(zip_path)
    click.secho(f"zip File size: {zip_path.stat().st_size / 1024:.2f} KiB")
    click.secho(f"Done. You'll find all pretrained weights in {zip_path}.", fg="green")


if __name__ == "__main__":
    main()
