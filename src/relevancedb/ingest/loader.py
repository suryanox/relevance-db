from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


SUPPORTED_FORMATS = {".txt", ".md", ".markdown"}


@dataclass
class Document:
    """Raw text from a single file, before chunking or enrichment."""

    path: Path
    text: str
    format: str 
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:80].replace("\n", " ")
        return (
            f"Document(path={self.path.name!r}, "
            f"format={self.format!r}, "
            f"chars={len(self.text)}, "
            f"preview={preview!r})"
        )


def load(path: str | Path) -> Document:
    """
    Load a single TXT or MD file.

    Raises:
        FileNotFoundError: File does not exist.
        ValueError: Format not supported yet.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format: {suffix!r}. "
            f"Currently supported: {sorted(SUPPORTED_FORMATS)}. "
            "PDF and DOCX support coming soon."
        )

    text = path.read_text(encoding="utf-8", errors="replace").strip()
    fmt = "md" if suffix in {".md", ".markdown"} else "txt"

    return Document(path=path, text=text, format=fmt)


def load_dir(path: str | Path, recursive: bool = True) -> list[Document]:
    """
    Load all supported files from a directory.

    Skips unsupported formats silently. Warns on unreadable files.
    """
    path = Path(path)

    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    pattern = "**/*" if recursive else "*"
    files = [
        f for f in path.glob(pattern)
        if f.suffix.lower() in SUPPORTED_FORMATS and f.is_file()
    ]

    docs = []
    for f in sorted(files):
        try:
            docs.append(load(f))
        except Exception as exc:
            print(f"[relevancedb] warning: skipping {f.name}: {exc}")

    return docs