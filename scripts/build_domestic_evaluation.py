"""Recreate domestic yearly/league performance and calibration from pregame predictions."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
GAMES = DATA / "lol_match_predictions_final.csv"
MODELS = {"V1": "v1_blue_prob", "V2": "v2_blue_prob", "V3": "v3_blue_prob"}


def performance(group, probability_column):
    y = group["blue_win"].astype(int).to_numpy()
    p = group[probability_column].astype(float).to_numpy()
    pc = np.clip(p, 1e-15, 1 - 1e-15)
    return {"Matches": len(group), "Accuracy": np.mean((p >= 0.5) == y),
            "Brier Score": np.mean((p - y) ** 2),
            "Log Loss": np.mean(-y * np.log(pc) - (1 - y) * np.log(1 - pc))}


def main():
    games = pd.read_csv(GAMES)
    rows, calibration = [], []
    for year in sorted(games["year"].unique()):
        year_games = games[games["year"] == year]
        for league in ["All", *sorted(year_games["league"].unique())]:
            subset = year_games if league == "All" else year_games[year_games["league"] == league]
            for model, prob in MODELS.items():
                rows.append({"Year": year, "League": league, "Model": model, **performance(subset, prob)})
                temp = subset[[prob, "blue_win"]].dropna().copy()
                temp["bucket"] = pd.qcut(temp[prob], min(10, len(temp)), duplicates="drop", labels=False)
                for bucket, group in temp.groupby("bucket"):
                    calibration.append({"Year": year, "League": league, "Model": model,
                                        "Bin": int(bucket), "Games": len(group),
                                        "Mean Predicted": group[prob].mean(), "Observed Win Rate": group["blue_win"].mean()})
    pd.DataFrame(rows).to_csv(DATA / "domestic_model_performance.csv", index=False)
    pd.DataFrame(calibration).to_csv(DATA / "domestic_calibration.csv", index=False)
    print("Domestic evaluation outputs regenerated")


if __name__ == "__main__":
    main()
