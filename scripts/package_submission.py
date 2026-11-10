"""Create the shareable submission archive without raw data or caches."""

from __future__ import annotations

import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "TAWOS_NLP_Problem10_Submission.zip"
EXCLUDED_DIRECTORIES = {"21308124", ".git", ".venv", "__pycache__", "artifacts", "checkpoints"}
EXCLUDED_FILES = {OUTPUT.name}


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return not any(part in EXCLUDED_DIRECTORIES for part in relative.parts) and path.name not in EXCLUDED_FILES and path.suffix != ".pyc"


def build() -> None:
    files = sorted(path for path in ROOT.rglob("*") if path.is_file() and included(path))
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT))
    print(f"Wrote {OUTPUT} with {len(files)} files ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build()

