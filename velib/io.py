"""Reading and writing raw snapshot files.

Two raw formats are supported, both holding records in the Paris Open Data
schema (see `config.RECORD_FIELDS`):

- `*.json`: one JSON array per file (the historical monthly exports).
- `*.jsonl`: one record per line (written by `velib fetch`, append-only).
  Finished days may be compressed to `*.jsonl.gz`.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from velib.config import RECORD_FIELDS

logger = logging.getLogger(__name__)

RAW_SUFFIXES = (".json", ".jsonl", ".jsonl.gz")


def iter_raw_files(dirs: Iterable[Path]) -> Iterator[Path]:
    """Yield every raw snapshot file under the given directories, sorted by name."""
    for directory in dirs:
        directory = Path(directory)
        if not directory.is_dir():
            logger.debug("Skipping missing input directory %s", directory)
            continue
        yield from sorted(
            p for p in directory.rglob("*") if p.name.endswith(RAW_SUFFIXES) and p.is_file()
        )


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read one raw file into a list of records."""
    path = Path(path)
    if path.name.endswith((".jsonl", ".jsonl.gz")):
        records = []
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    # A truncated last line (e.g. fetcher killed mid-write) must
                    # not make the whole day unreadable.
                    logger.warning("Skipping malformed line %d in %s", line_number, path)
        return records

    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array of records")
    return data


def records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert records to a DataFrame holding only the columns the pipeline uses."""
    frame = pd.DataFrame.from_records(records)
    for column in RECORD_FIELDS:
        if column not in frame.columns:
            frame[column] = None
    return frame[RECORD_FIELDS]


def load_raw(dirs: Iterable[Path]) -> pd.DataFrame:
    """Load every raw file under `dirs` into one DataFrame."""
    dirs = list(dirs)
    frames = []
    for path in iter_raw_files(dirs):
        records = read_records(path)
        if records:
            frames.append(records_to_frame(records))
        logger.debug("Read %d records from %s", len(records), path)

    if not frames:
        searched = ", ".join(str(d) for d in dirs)
        raise FileNotFoundError(
            f"No raw snapshot files found in: {searched}. "
            "Run `python -m velib fetch` to collect data, or `python -m velib demo` "
            "to try the pipeline on synthetic data."
        )

    frame = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d raw records from %d files", len(frame), len(frames))
    return frame


def append_jsonl(records: Iterable[dict[str, Any]], path: Path) -> int:
    """Append records to a JSONL file, creating it if needed. Returns the count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json_atomic(data: Any, path: Path, indent: int | None = None) -> None:
    """Write JSON so that readers never observe a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        os.chmod(tmp_name, 0o644)  # mkstemp creates 0600; the map server must read it
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
