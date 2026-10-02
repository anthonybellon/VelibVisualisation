"""Collect live station snapshots from the Paris Open Data API or the Vélib' GBFS feed.

Both sources are normalised to the same record schema (`config.RECORD_FIELDS`)
plus a `fetched_at` timestamp, and appended to one JSONL file per UTC day.
"""

from __future__ import annotations

import gzip
import json
import logging
import time
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from velib.config import (
    GBFS_INFO_URL,
    GBFS_STATUS_URL,
    HTTP_TIMEOUT_SECONDS,
    OPENDATA_URL,
)
from velib.io import append_jsonl

logger = logging.getLogger(__name__)

USER_AGENT = "VelibVisualisation/2.0"


def get_json(url: str, timeout: float = HTTP_TIMEOUT_SECONDS) -> Any:
    """GET an https URL and decode its JSON body (transparently handling gzip)."""
    if not url.startswith("https://"):
        raise ValueError(f"Refusing to fetch non-https URL: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        body = response.read()
    if body[:2] == b"\x1f\x8b":
        body = gzip.decompress(body)
    return json.loads(body)


def _yes_no(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return "OUI" if value else "NON"


def normalize_opendata_record(record: dict[str, Any]) -> dict[str, Any]:
    """Normalise one Paris Open Data record.

    The v2.1 API returns flat records with `coordonnees_geo = {lat, lon}`. The
    legacy v1 export wrapped them in `fields` with `coordonnees_geo = [lat, lon]`;
    both are accepted so older dumps can still be loaded.
    """
    fields = record.get("fields", record)
    geo = fields.get("coordonnees_geo")
    if isinstance(geo, (list, tuple)) and len(geo) == 2:
        geo = {"lat": geo[0], "lon": geo[1]}
    return {
        "stationcode": fields.get("stationcode"),
        "name": fields.get("name"),
        "is_installed": fields.get("is_installed"),
        "is_renting": fields.get("is_renting"),
        "is_returning": fields.get("is_returning"),
        "capacity": fields.get("capacity"),
        "numbikesavailable": fields.get("numbikesavailable"),
        "numdocksavailable": fields.get("numdocksavailable"),
        "mechanical": fields.get("mechanical"),
        "ebike": fields.get("ebike"),
        "duedate": fields.get("duedate"),
        "coordonnees_geo": geo,
    }


def normalize_gbfs(status: dict[str, Any], info: dict[str, Any]) -> list[dict[str, Any]]:
    """Join GBFS `station_status` and `station_information` into Open Data records."""
    info_by_id = {s["station_id"]: s for s in info["data"]["stations"]}
    records = []
    for st in status["data"]["stations"]:
        meta = info_by_id.get(st["station_id"])
        if meta is None:
            continue
        bike_types: dict[str, int] = {}
        for entry in st.get("num_bikes_available_types", []):
            bike_types.update(entry)
        last_reported = st.get("last_reported")
        duedate = (
            datetime.fromtimestamp(last_reported, tz=timezone.utc).isoformat()
            if last_reported
            else None
        )
        records.append(
            {
                "stationcode": str(meta.get("stationCode", st.get("stationCode"))),
                "name": meta.get("name"),
                "is_installed": _yes_no(st.get("is_installed")),
                "is_renting": _yes_no(st.get("is_renting")),
                "is_returning": _yes_no(st.get("is_returning")),
                "capacity": meta.get("capacity"),
                "numbikesavailable": st.get("num_bikes_available"),
                "numdocksavailable": st.get("num_docks_available"),
                "mechanical": bike_types.get("mechanical"),
                "ebike": bike_types.get("ebike"),
                "duedate": duedate,
                "coordonnees_geo": {"lat": meta.get("lat"), "lon": meta.get("lon")},
            }
        )
    return records


def fetch_opendata() -> list[dict[str, Any]]:
    data = get_json(OPENDATA_URL)
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    return [normalize_opendata_record(r) for r in data]


def fetch_gbfs() -> list[dict[str, Any]]:
    return normalize_gbfs(get_json(GBFS_STATUS_URL), get_json(GBFS_INFO_URL))


SOURCES: dict[str, Callable[[], list[dict[str, Any]]]] = {
    "opendata": fetch_opendata,
    "gbfs": fetch_gbfs,
}


def fetch_snapshot(source: str = "opendata", now: datetime | None = None) -> list[dict[str, Any]]:
    """Fetch one snapshot of every station and stamp it with `fetched_at`."""
    now = now or datetime.now(timezone.utc)
    records = SOURCES[source]()
    fetched_at = now.isoformat(timespec="seconds")
    for record in records:
        record["fetched_at"] = fetched_at
    return records


def save_snapshot(records: list[dict[str, Any]], raw_dir: Path, now: datetime) -> Path:
    path = Path(raw_dir) / f"{now:%Y-%m-%d}.jsonl"
    count = append_jsonl(records, path)
    logger.info("Appended %d records to %s", count, path)
    return path


def run_fetch(raw_dir: Path, source: str = "opendata", interval_minutes: float | None = None):
    """Fetch once, or forever every `interval_minutes` (errors are logged, not fatal)."""
    while True:
        now = datetime.now(timezone.utc)
        try:
            save_snapshot(fetch_snapshot(source, now), raw_dir, now)
        except Exception:
            if interval_minutes is None:
                raise
            logger.exception("Fetch failed; retrying at the next interval")
        if interval_minutes is None:
            return
        time.sleep(interval_minutes * 60)
