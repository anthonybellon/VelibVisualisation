# Vélib' availability forecast

Forecasts bike availability at every Vélib' station in Paris for each hour of the
coming week, and writes it as JSON for a front end. This repository also contains a
small **model debugging dashboard** for finding where the forecast is wrong.

How it works, the data contracts and the current status are in
**[ARCHITECTURE.md](ARCHITECTURE.md)**.

## Quick start

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: .\venv\Scripts\activate
pip install -e ".[dev]"

# No data yet? Run the whole pipeline on synthetic data:
python -m velib demo
python -m velib serve             # open http://127.0.0.1:8000
```

Requires Python 3.10+.

## Real data

```bash
# 1. Collect snapshots (one now, or every 15 minutes until stopped).
#    Training needs at least ~3 weeks: 1 week of history + training days + a 7-day holdout.
python -m velib fetch
python -m velib fetch --loop               # every 15 min; or schedule `velib fetch` with cron/launchd

# Older data is picked up too: monthly dumps in data/historical_data_cleaned/*.json,
# and snapshots saved by the v1 fetcher (copy them into data/raw/; duplicates are removed).

# 2. Prepare, train, forecast
python -m velib run                        # = prepare + train + forecast
```

Outputs, all under `data/` (git-ignored):

| File | What |
|---|---|
| `4_compressed_predictions_final_fix.json` | **Front-end export** (same schema as v1; see ARCHITECTURE.md §6.6) |
| `forecast.json` | Next-week forecast for the dashboard |
| `models/metrics.json` | Holdout MAE/RMSE for the model and both baselines |
| `models/diagnostics.json` | Per-station / per-hour errors, series, feature importance |
| `models/model.joblib` | Trained model bundle |

`python -m velib --help` lists every command and option. `--data-dir DIR` points any
command at another data directory.

## Debugging dashboard

`python -m velib serve` serves `index.html`, which reads `data/` (or `data/demo/` if
there is no real run yet; choose explicitly with `?dir=data/demo`). It shows:

- holdout error for the model against "same hour last week" and "weekly average";
- error by hour of day and day of week, and permutation feature importance;
- a map coloured by model error, skill vs baseline, data coverage or forecast fill;
- the worst stations, and for any station its actual vs predicted holdout week and
  its next-week forecast (shareable as `#station=<code>`).

## Development

```bash
pytest                                     # runs on synthetic data, no network needed
ruff check . && ruff format --check .
pre-commit install                         # optional
```

CI runs lint, tests and a demo pipeline on Python 3.10 and 3.13.

## Project layout

```
velib/            the package (fetch, preprocessing, features, model, forecast, CLI)
tests/            pytest suite (synthetic data)
index.html, web/  debugging dashboard
sample_data/      one real snapshot (station locations/capacities for synthetic data)
archive/          one-off v1 scripts, for reference only
notebooks/        v1 exploration notebooks (stale: they import removed modules)
```

## License

See [LICENSE](LICENSE).
