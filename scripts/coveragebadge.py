#!/usr/bin/env python3
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

COVERAGE_XML = "coverage.xml"
INPUT_SVG = "artwork/_coverage_template.svg"
OUTPUT_SVG = "artwork/coverage.svg"

PLACEHOLDER_COLOR = "#cba317"


def get_coverage():
    tree = ET.parse(COVERAGE_XML)
    root = tree.getroot()

    line_rate = root.get("line-rate")
    if line_rate is None:
        raise RuntimeError("line-rate attribute not found in coverage.xml")

    percent = float(line_rate) * 100.0
    return percent, f"{percent:.2f}%"


def get_badge_color(percent):
    if percent >= 75:
        return "#40c010"
    elif percent >= 50:
        return "#dfb317"
    else:
        return "#e05d44"


def main():
    subprocess.run("uv run coverage run --source ncalab -m pytest tests".split(" "))
    subprocess.run("uv run coverage xml".split(" "))

    percent, coverage_text = get_coverage()
    color = get_badge_color(percent)

    svg = Path(INPUT_SVG).read_text(encoding="utf-8")
    svg = svg.replace("xx.xx%", coverage_text)
    svg = svg.replace(f'fill="{PLACEHOLDER_COLOR}"', f'fill="{color}"')

    Path(OUTPUT_SVG).write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
