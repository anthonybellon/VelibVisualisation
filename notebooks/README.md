# Notebooks (v1, stale)

These notebooks were written against the v1 `scripts/` modules, which have been
replaced by the `velib` package. They import functions that no longer exist and
will not run as-is. They are kept for their exploratory analysis.

To explore v2 data in a notebook:

```python
from velib.config import Paths, DEFAULT_DATA_DIR
from velib.preprocessing import HourlyData
from velib.features import build_features

hourly = HourlyData.load(Paths(DEFAULT_DATA_DIR).hourly_path)
frame = build_features(hourly)
```
