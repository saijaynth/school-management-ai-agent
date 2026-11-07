"""Package the project into a clean zip distribution file for email sharing."""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ZIP = ROOT / "school-management-ai-agent.zip"

EXCLUDE_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    "_pgportable",
    ".freebuff",
    ".gemini",
}

EXCLUDE_EXTS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".safetensors",
    ".bin",
    ".pt",
    ".log",
    ".tsbuildinfo",
}

def make_zip():
    print(f"Creating zip archive at {OUTPUT_ZIP}...")
    file_count = 0

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in ROOT.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(ROOT)

            # Check exclusions
            parts = rel.parts
            if any(part in EXCLUDE_DIRS for part in parts):
                continue
            if p.suffix in EXCLUDE_EXTS:
                continue
            if p.name == "school-management-ai-agent.zip":
                continue

            # Add to zip
            zf.write(p, rel)
            file_count += 1

    size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"Zip created successfully: {file_count} files, {size_mb:.2f} MB")
    return OUTPUT_ZIP, size_mb

if __name__ == "__main__":
    make_zip()
