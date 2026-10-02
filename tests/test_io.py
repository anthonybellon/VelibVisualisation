import gzip
import json

import pytest

from velib.config import RECORD_FIELDS
from velib.io import (
    append_jsonl,
    iter_raw_files,
    load_raw,
    read_records,
    records_to_frame,
    write_json_atomic,
)


def test_reads_json_array_jsonl_and_gzip(tmp_path, record):
    (tmp_path / "a.json").write_text(json.dumps([record("1"), record("2")]))
    append_jsonl([record("3")], tmp_path / "b.jsonl")
    with gzip.open(tmp_path / "c.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write(json.dumps(record("4")) + "\n")
    (tmp_path / "ignored.txt").write_text("not data")

    files = list(iter_raw_files([tmp_path]))
    assert [p.name for p in files] == ["a.json", "b.jsonl", "c.jsonl.gz"]

    frame = load_raw([tmp_path])
    assert sorted(frame["stationcode"]) == ["1", "2", "3", "4"]
    assert list(frame.columns) == RECORD_FIELDS


def test_truncated_jsonl_line_is_skipped(tmp_path, record):
    path = tmp_path / "day.jsonl"
    path.write_text(json.dumps(record("1")) + "\n" + '{"stationcode": "2", "na')
    assert [r["stationcode"] for r in read_records(path)] == ["1"]


def test_json_must_be_an_array(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{}")
    with pytest.raises(ValueError):
        read_records(path)


def test_load_raw_without_files_explains_what_to_do(tmp_path):
    with pytest.raises(FileNotFoundError, match="velib demo"):
        load_raw([tmp_path / "missing"])


def test_records_to_frame_adds_missing_columns(record):
    rec = record()
    del rec["ebike"]
    rec["unused"] = 1
    frame = records_to_frame([rec])
    assert list(frame.columns) == RECORD_FIELDS
    assert frame["ebike"].isna().all()


def test_write_json_atomic_is_readable_and_leaves_no_temp_files(tmp_path):
    path = tmp_path / "out" / "doc.json"
    write_json_atomic({"a": 1}, path)
    assert json.loads(path.read_text()) == {"a": 1}
    assert oct(path.stat().st_mode & 0o777) == "0o644"
    assert [p.name for p in path.parent.iterdir()] == ["doc.json"]
