"""Data integrity checks for the LoL Elo portfolio.

Place this file at tests/test_data_quality.py in the repository.
Checks fail loudly if exported model data changes unexpectedly.
"""
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_csv(filename):
    path = ROOT / filename
    assert path.is_file(), f"Missing required data file: {filename}"
    return pd.read_csv(path)


def test_domestic_required_columns():
    df = load_csv("lol_match_predictions_final.csv")
    required = {"gameid", "date", "year", "league", "blue_win", "v1_blue_prob", "v2_blue_prob", "v3_blue_prob"}
    assert required.issubset(df.columns), f"Missing: {required - set(df.columns)}"
    assert len(df) > 0


def test_v4_required_columns():
    df = load_csv("v4_match_predictions_final.csv")
    required = {"gameid", "date", "year", "blue_win", "blue_team", "red_team", "cross_region", "v4_blue_prob", "v4_adjusted_prob"}
    assert required.issubset(df.columns), f"Missing: {required - set(df.columns)}"
    assert len(df) > 0


@pytest.mark.parametrize("filename", ["lol_match_predictions_final.csv", "v4_match_predictions_final.csv"])
def test_unique_game_ids(filename):
    df = load_csv(filename)
    assert df["gameid"].notna().all()
    assert not df["gameid"].duplicated().any(), f"Duplicate game IDs in {filename}"


@pytest.mark.parametrize("filename", ["lol_match_predictions_final.csv", "v4_match_predictions_final.csv"])
def test_valid_dates_and_results(filename):
    df = load_csv(filename)
    dates = pd.to_datetime(df["date"], errors="coerce")
    assert dates.notna().all(), f"Invalid dates in {filename}"
    assert df["blue_win"].notna().all()
    assert df["blue_win"].isin([0, 1]).all(), f"Invalid binary results in {filename}"
    assert (dates.dt.year == df["year"]).all()


def test_domestic_probabilities():
    df = load_csv("lol_match_predictions_final.csv")
    for col in ("v1_blue_prob", "v2_blue_prob", "v3_blue_prob"):
        assert df[col].notna().all()
        assert df[col].between(0, 1).all(), f"Invalid probabilities in {col}"


def test_v4_probabilities_and_elo():
    df = load_csv("v4_match_predictions_final.csv")
    for col in ("v4_blue_prob", "v4_adjusted_prob"):
        assert df[col].notna().all()
        assert df[col].between(0, 1).all(), f"Invalid probabilities in {col}"
    for col in ("v4_blue_elo", "v4_red_elo"):
        assert pd.to_numeric(df[col], errors="coerce").notna().all()


def test_rating_snapshots():
    for filename, rating_col, key_cols in (
        ("latest_team_ratings.csv", "elo_rating", ["league", "team"]),
        ("international_team_ratings.csv", "elo", ["team"]),
    ):
        df = load_csv(filename)
        assert set(key_cols + [rating_col]).issubset(df.columns)
        assert df[key_cols].notna().all().all()
        assert not df.duplicated(key_cols).any(), f"Duplicate teams in {filename}"
        assert pd.to_numeric(df[rating_col], errors="coerce").notna().all()


def test_series_data_consistency():
    df = load_csv("series_verified_backtest.csv")
    required = {"series_id", "best_of", "games", "score", "series_prob_a", "result_a"}
    assert required.issubset(df.columns)
    assert not df["series_id"].duplicated().any()
    assert df["best_of"].isin([3, 5]).all()
    assert df["games"].between(2, 5).all()
    assert df["series_prob_a"].notna().all() and df["series_prob_a"].between(0, 1).all()
    assert df["result_a"].isin([0, 1]).all()
    for row in df.itertuples(index=False):
        a, b = map(int, row.score.split("-"))
        needed = row.best_of // 2 + 1
        assert max(a, b) == needed and min(a, b) < needed
        assert a + b == row.games
        assert row.result_a == int(a > b)
