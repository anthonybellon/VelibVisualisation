"""Command line interface: `python -m velib <command>`.

fetch     collect live snapshots into <data-dir>/raw/
prepare   raw snapshots -> hourly grid (<data-dir>/processed/hourly.pkl)
train     hourly grid -> model + holdout metrics (<data-dir>/models/)
forecast  model + hourly grid -> next-week forecast for the map (<data-dir>/forecast.json)
run       prepare + train + forecast
demo      generate synthetic data into data/demo/ and run the whole pipeline on it
serve     serve the map at http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import functools
import http.server
import logging
import shutil
import sys
from pathlib import Path

from velib.config import (
    DEFAULT_DATA_DIR,
    DEFAULT_FETCH_INTERVAL_MINUTES,
    DEFAULT_TEST_DAYS,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    PROJECT_ROOT,
    Paths,
)

logger = logging.getLogger("velib")


def init_logging(verbose: bool = False, quiet: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT, force=True)


# ------------------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------------------


def cmd_fetch(paths: Paths, args: argparse.Namespace) -> None:
    from velib.fetch import run_fetch

    run_fetch(paths.raw_dir, source=args.source, interval_minutes=args.loop)


def cmd_prepare(paths: Paths, args: argparse.Namespace) -> None:
    from velib.io import load_raw
    from velib.preprocessing import clean_records, to_hourly

    hourly = to_hourly(clean_records(load_raw(paths.input_dirs())))
    hourly.save(paths.hourly_path)


def cmd_train(paths: Paths, args: argparse.Namespace) -> None:
    from velib.io import write_json_atomic
    from velib.model import save_bundle, train
    from velib.preprocessing import HourlyData

    bundle, diagnostics = train(
        HourlyData.load(paths.hourly_path), test_days=args.test_days, refit=not args.no_refit
    )
    save_bundle(bundle, paths.model_path)
    report = {k: v for k, v in bundle.items() if k != "model"}
    write_json_atomic(report, paths.metrics_path, indent=2)
    write_json_atomic(diagnostics, paths.diagnostics_path)
    logger.info("Wrote metrics and diagnostics to %s", paths.metrics_path.parent)


def cmd_forecast(paths: Paths, args: argparse.Namespace) -> None:
    from velib.forecast import export_forecast, export_legacy, forecast
    from velib.io import write_json_atomic
    from velib.model import load_bundle
    from velib.preprocessing import HourlyData

    hourly = HourlyData.load(paths.hourly_path)
    bundle = load_bundle(paths.model_path)
    predictions = forecast(hourly, bundle)
    document = export_forecast(
        predictions, hourly, bundle, synthetic=getattr(args, "synthetic", False)
    )
    write_json_atomic(document, paths.forecast_path)
    write_json_atomic(export_legacy(predictions, hourly), paths.legacy_export_path)
    logger.info("Wrote front-end export to %s", paths.legacy_export_path)
    logger.info(
        "Wrote forecast for %d stations to %s", len(document["stations"]), paths.forecast_path
    )


def cmd_run(paths: Paths, args: argparse.Namespace) -> None:
    cmd_prepare(paths, args)
    cmd_train(paths, args)
    cmd_forecast(paths, args)


def cmd_demo(paths: Paths, args: argparse.Namespace) -> None:
    from velib.synthetic import generate_records, write_records

    if paths.raw_dir.exists():
        shutil.rmtree(paths.raw_dir)
    records = generate_records(
        n_stations=args.stations, days=args.days, start=args.start, seed=args.seed
    )
    write_records(records, paths.raw_dir)
    logger.info("Generated %d synthetic records in %s", len(records), paths.raw_dir)
    args.synthetic = True
    cmd_run(paths, args)
    logger.info("Demo ready. Run `python -m velib serve` and open %s", _dashboard_url(8000, paths))


def cmd_serve(paths: Paths, args: argparse.Namespace) -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(PROJECT_ROOT))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    if not paths.diagnostics_path.exists():
        print(f"Note: {paths.diagnostics_path} does not exist yet; run `train` or `demo` first.")
    print(f"Serving the dashboard at {_dashboard_url(args.port, paths)} (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _dashboard_url(port: int, paths: Paths) -> str:
    """URL of the dashboard showing `paths.data_dir` (must live inside the project)."""
    url = f"http://127.0.0.1:{port}/"
    try:
        rel = Path(paths.data_dir).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return url + "  (warning: data dir is outside the project, so the server cannot see it)"
    return url if rel == "data" else f"{url}?dir={rel}"


# ------------------------------------------------------------------------------
# Parser
# ------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m velib",
        description="Vélib' availability forecasting pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n", 1)[1],
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=f"Data directory (default: {DEFAULT_DATA_DIR}, or data/demo for `demo`)",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    verbosity.add_argument("-q", "--quiet", action="store_true", help="Warnings only")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    p = sub.add_parser("fetch", help="Collect live station snapshots")
    p.add_argument("--source", choices=["opendata", "gbfs"], default="opendata")
    p.add_argument(
        "--loop",
        type=float,
        nargs="?",
        const=DEFAULT_FETCH_INTERVAL_MINUTES,
        default=None,
        metavar="MINUTES",
        help=f"Keep fetching every MINUTES (default {DEFAULT_FETCH_INTERVAL_MINUTES})",
    )
    p.set_defaults(func=cmd_fetch)

    sub.add_parser("prepare", help="Build the hourly grid").set_defaults(func=cmd_prepare)

    def add_train_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--test-days", type=int, default=DEFAULT_TEST_DAYS, help="Holdout length")
        p.add_argument(
            "--no-refit",
            action="store_true",
            help="Keep the model trained without the holdout instead of refitting on all data",
        )

    p = sub.add_parser("train", help="Train and evaluate the model")
    add_train_args(p)
    p.set_defaults(func=cmd_train)

    sub.add_parser("forecast", help="Forecast the next week").set_defaults(func=cmd_forecast)

    p = sub.add_parser("run", help="prepare + train + forecast")
    add_train_args(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("demo", help="Run the pipeline on synthetic data")
    p.add_argument("--stations", type=int, default=250)
    p.add_argument("--days", type=int, default=35)
    p.add_argument("--start", default="2024-03-04", help="First UTC day (YYYY-MM-DD)")
    p.add_argument("--seed", type=int, default=0)
    add_train_args(p)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("serve", help="Serve the debugging dashboard locally")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    init_logging(args.verbose, args.quiet)
    data_dir = args.data_dir or (
        DEFAULT_DATA_DIR / "demo" if args.command == "demo" else DEFAULT_DATA_DIR
    )
    try:
        args.func(Paths(data_dir), args)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
