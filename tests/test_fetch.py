from datetime import datetime, timezone

from velib import fetch
from velib.io import read_records


def test_normalizes_current_opendata_record(record):
    rec = record("16107", extra_field="dropped")
    out = fetch.normalize_opendata_record(rec)
    assert out["stationcode"] == "16107"
    assert out["coordonnees_geo"] == {"lat": 48.86, "lon": 2.35}
    assert "extra_field" not in out


def test_normalizes_legacy_v1_record_with_coordinate_list():
    legacy = {
        "fields": {
            "stationcode": "9020",
            "numbikesavailable": 4,
            "capacity": 21,
            "duedate": "2024-05-28T21:02:36+00:00",
            "coordonnees_geo": [48.879, 2.337],
        }
    }
    out = fetch.normalize_opendata_record(legacy)
    assert out["coordonnees_geo"] == {"lat": 48.879, "lon": 2.337}
    assert out["numbikesavailable"] == 4


def test_normalizes_gbfs_feeds():
    status = {
        "data": {
            "stations": [
                {
                    "station_id": 1,
                    "num_bikes_available": 23,
                    "num_bikes_available_types": [{"mechanical": 8}, {"ebike": 15}],
                    "num_docks_available": 11,
                    "is_installed": 1,
                    "is_renting": 0,
                    "is_returning": 1,
                    "last_reported": 1716930000,
                },
                {"station_id": 999, "num_bikes_available": 1},  # no station info: skipped
            ]
        }
    }
    info = {
        "data": {
            "stations": [
                {
                    "station_id": 1,
                    "stationCode": "16107",
                    "name": "Benjamin Godard",
                    "lat": 48.86,
                    "lon": 2.27,
                    "capacity": 35,
                }
            ]
        }
    }
    [out] = fetch.normalize_gbfs(status, info)
    assert out["stationcode"] == "16107"
    assert out["numbikesavailable"] == 23
    assert (out["mechanical"], out["ebike"]) == (8, 15)
    assert (out["is_installed"], out["is_renting"]) == ("OUI", "NON")
    assert out["duedate"] == "2024-05-28T21:00:00+00:00"
    assert out["coordonnees_geo"] == {"lat": 48.86, "lon": 2.27}


def test_fetch_appends_to_one_file_per_utc_day(tmp_path, monkeypatch, record):
    monkeypatch.setitem(fetch.SOURCES, "opendata", lambda: [record("1"), record("2")])
    now = datetime(2024, 5, 28, 23, 30, tzinfo=timezone.utc)
    for _ in range(2):
        fetch.save_snapshot(fetch.fetch_snapshot("opendata", now), tmp_path, now)

    records = read_records(tmp_path / "2024-05-28.jsonl")
    assert len(records) == 4
    assert {r["fetched_at"] for r in records} == {"2024-05-28T23:30:00+00:00"}
