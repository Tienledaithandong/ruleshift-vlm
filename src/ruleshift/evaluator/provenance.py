"""Hash canonical research sources while excluding generated runs and staged copies."""

import hashlib
import json
from pathlib import Path

HASH_SUFFIXES = {".py", ".txt", ".yaml", ".yml", ".csv", ".md", ".toml", ".lock", ".ipynb", ".json"}
SOURCE_DIRS = ("src/ruleshift", "scripts", "configs", "tests", "research", "docs",
               "kaggle/kernel", "notebooks", "prototypes", ".github/workflows")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_digest() -> str:
    """Hash the installed implementation and prompts independent of checkout line endings."""
    package = Path(__file__).resolve().parents[1]
    hashes = {}
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".txt"}:
            content = path.read_text(encoding="utf-8").encode("utf-8")
            hashes[path.relative_to(package).as_posix()] = hashlib.sha256(content).hexdigest()
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode("utf-8")).hexdigest()


def collect_hashes(root: Path) -> dict[str, str]:
    paths = [root / name for name in ("pyproject.toml", "requirements.lock", "README.md")]
    for directory in SOURCE_DIRS:
        paths.extend((root / directory).rglob("*"))
    return {
        path.relative_to(root).as_posix(): file_hash(path)
        for path in sorted(paths) if path.is_file() and path.suffix in HASH_SUFFIXES
    }
