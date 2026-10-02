"""Spatial neighbourhoods between stations.

Distances are great-circle (haversine) distances in metres. The previous
implementation converted metres to degrees with 111.32 km/degree on both axes,
which is wrong for longitude at Paris' latitude (~73 km/degree) and turned the
search circle into an ellipse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import BallTree

from velib.config import NEIGHBOUR_COUNT, NEIGHBOUR_MAX_RADIUS_M

EARTH_RADIUS_M = 6_371_000


def neighbour_adjacency(
    stations: pd.DataFrame,
    k: int = NEIGHBOUR_COUNT,
    max_radius_m: float = NEIGHBOUR_MAX_RADIUS_M,
) -> sparse.csr_matrix:
    """Binary adjacency matrix: row i marks up to `k` nearest stations within `max_radius_m`.

    Rows and columns follow the order of `stations`. A station is never its own
    neighbour. Stations without coordinates have no neighbours.
    """
    n = len(stations)
    coords = stations[["lat", "lon"]].to_numpy(dtype=float)
    valid = np.flatnonzero(~np.isnan(coords).any(axis=1))
    rows: list[int] = []
    cols: list[int] = []

    if len(valid) > 1 and k > 0:
        tree = BallTree(np.radians(coords[valid]), metric="haversine")
        n_query = min(k + 1, len(valid))
        distances, indices = tree.query(np.radians(coords[valid]), k=n_query)
        distances_m = distances * EARTH_RADIUS_M
        for i, (dist_row, idx_row) in enumerate(zip(distances_m, indices, strict=True)):
            chosen = [
                valid[j]
                for d, j in zip(dist_row, idx_row, strict=True)
                if j != i and d <= max_radius_m
            ][:k]
            rows.extend([valid[i]] * len(chosen))
            cols.extend(chosen)

    data = np.ones(len(rows), dtype=float)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


def neighbour_mean(values: pd.DataFrame, adjacency: sparse.csr_matrix) -> pd.DataFrame:
    """Mean of each station's neighbours' values at each time, ignoring missing values.

    `values` is a time x station frame whose columns follow the adjacency order.
    The result is NaN where no neighbour has a value.
    """
    array = values.to_numpy(dtype=float)
    observed = ~np.isnan(array)
    totals = adjacency @ np.where(observed, array, 0.0).T
    counts = adjacency @ observed.T.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(counts > 0, totals / counts, np.nan)
    return pd.DataFrame(mean.T, index=values.index, columns=values.columns)
