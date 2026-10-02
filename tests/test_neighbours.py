import numpy as np
import pandas as pd
import pytest

from velib.neighbours import EARTH_RADIUS_M, neighbour_adjacency, neighbour_mean

LAT0, LON0 = 48.86, 2.35
M_PER_DEG_LAT = np.pi * EARTH_RADIUS_M / 180
M_PER_DEG_LON = M_PER_DEG_LAT * np.cos(np.radians(LAT0))


def _stations(offsets_m):
    """Stations at (north, east) metre offsets from a Paris origin."""
    return pd.DataFrame(
        {
            "lat": [LAT0 + n / M_PER_DEG_LAT for n, _ in offsets_m],
            "lon": [LON0 + e / M_PER_DEG_LON for _, e in offsets_m],
        },
        index=[str(i) for i in range(len(offsets_m))],
    )


def _neighbours(adjacency, i):
    return set(adjacency[i].indices.tolist())


def test_radius_is_a_circle_in_metres_not_degrees():
    # 600 m north and 600 m east are both inside 700 m. The old degree-based
    # conversion measured the eastern one as ~910 m and dropped it.
    stations = _stations([(0, 0), (600, 0), (0, 600), (0, 800)])
    adjacency = neighbour_adjacency(stations, k=5, max_radius_m=700)
    assert _neighbours(adjacency, 0) == {1, 2}


def test_never_its_own_neighbour_even_with_duplicate_coordinates():
    stations = _stations([(0, 0), (0, 0), (100, 0)])
    adjacency = neighbour_adjacency(stations, k=5, max_radius_m=500)
    for i in range(3):
        assert i not in _neighbours(adjacency, i)
    assert _neighbours(adjacency, 0) == {1, 2}


def test_keeps_only_the_k_nearest():
    stations = _stations([(0, 0), (100, 0), (200, 0), (300, 0), (400, 0)])
    adjacency = neighbour_adjacency(stations, k=2, max_radius_m=2000)
    assert _neighbours(adjacency, 0) == {1, 2}


def test_stations_without_coordinates_have_no_neighbours():
    stations = _stations([(0, 0), (100, 0)])
    stations.loc["2"] = [np.nan, np.nan]
    adjacency = neighbour_adjacency(stations, k=5, max_radius_m=500)
    assert _neighbours(adjacency, 2) == set()
    assert 2 not in _neighbours(adjacency, 0)


def test_neighbour_mean_ignores_missing_values():
    stations = _stations([(0, 0), (100, 0), (200, 0)])
    adjacency = neighbour_adjacency(stations, k=2, max_radius_m=500)
    values = pd.DataFrame({"0": [0.0, 0.0], "1": [0.2, np.nan], "2": [0.6, np.nan]})
    result = neighbour_mean(values, adjacency)
    assert result.loc[0, "0"] == pytest.approx(0.4)  # mean of stations 1 and 2
    assert np.isnan(result.loc[1, "0"])  # no neighbour observed
    assert result.loc[1, "1"] == 0.0  # only station 0 observed
