"""Build exploratory series backtest from audit candidates (not independent verification).

Uses first game's PRE-match Elo and a neutral-side, constant game win probability.
Audit rows are sourced from series_source_audit.csv; its links
are leads for manual checking and do not constitute row-by-row verification.
"""
from pathlib import Path
from math import comb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def series_win_probability(p, best_of):
    needed = best_of // 2 + 1
    return sum(comb(needed - 1 + losses, losses) * p**needed * (1-p)**losses
               for losses in range(needed))


def build():
    games = pd.read_csv(DATA / "v4_match_predictions_final.csv")
    audit = pd.read_csv(DATA / "series_source_audit.csv")
    ids = games["gameid"].astype(str).str.extract(r"^(?P<series_id>.+)_game_(?P<game_no>\d+)$")
    games = pd.concat([games, ids], axis=1)
    games["game_no"] = pd.to_numeric(games["game_no"], errors="coerce")
    rows = []
    for match in audit.itertuples(index=False):
        series_games = games[games.series_id == match.series_id].sort_values("game_no")
        if len(series_games) != int(match.games) or len(series_games) == 0:
            continue
        first = series_games.iloc[0]
        if match.team_a == first.blue_team:
            a, b = float(first.v4_blue_elo), float(first.v4_red_elo)
        elif match.team_a == first.red_team:
            a, b = float(first.v4_red_elo), float(first.v4_blue_elo)
        else:
            continue
        p = 1 / (1 + 10 ** ((b-a) / 400))
        best_of = int(match.best_of)
        if best_of not in (3, 5):
            continue
        rows.append({"series_id": match.series_id, "date": match.date, "year": match.year,
                     "competition": match.league, "team_a": match.team_a,
                     "team_b": match.team_b, "score": f"{match.wins_a}-{match.wins_b}",
                     "games": int(match.games), "best_of": best_of,
                     "pre_series_elo_a": a, "pre_series_elo_b": b,
                     "neutral_game_prob_a": p,
                     "series_prob_a": series_win_probability(p, best_of),
                     "result_a": int(match.result_a),
                     "source_link": str(match.reference),
                     "validation_status": "unverified_source_link"})
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "series_exploratory_backtest.csv", index=False)
    print(f"Wrote {len(out)} exploratory series; no row is labelled independently verified")


if __name__ == "__main__":
    build()
