"""Append-only pre-match LoL forward testing using frozen Elo snapshots.

Run `python forward_test.py` from the repository root. Commit the resulting
forward_test/ files to GitHub BEFORE the match starts to document timing.
No results or ratings are downloaded or updated by this script.
"""
import csv
import importlib.util
import hashlib
import json
import math
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TRACK = ROOT / "forward_test"
SNAPS = TRACK / "frozen"
MANIFEST = TRACK / "model_lock.json"
PREDS = TRACK / "predictions.csv"
RESULTS = TRACK / "results.csv"
RATINGS = {"V3": "latest_team_ratings.csv", "V4": "international_team_ratings.csv"}
PRED_FIELDS = ["prediction_id", "recorded_at_utc", "scheduled_at_utc", "model", "format", "team_a", "team_b", "team_a_region", "team_b_region", "elo_a", "elo_b", "blue_bonus", "p_game_team_a", "p_series_team_a", "model_lock_sha256"]
RESULT_FIELDS = ["prediction_id", "resolved_at_utc", "winner", "score_a", "score_b"]


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def append(path, fieldnames, values):
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(values)


def lock():
    TRACK.mkdir(exist_ok=True)
    SNAPS.mkdir(exist_ok=True)
    if not MANIFEST.exists():
        for filename in RATINGS.values():
            source = DATA / filename
            if not source.exists():
                raise FileNotFoundError(f"Missing data file: {source}")
            shutil.copy2(source, SNAPS / filename)
        model_source = ROOT / "elo_model.py"
        series_source = ROOT / "series_probability.py"
        for source in [model_source, series_source]:
            if not source.exists():
                raise FileNotFoundError(f"Missing model file: {source}")
            shutil.copy2(source, SNAPS / source.name)
        info = {"locked_at_utc": timestamp(), "description": "Frozen V3/V4 model CODE AND RATINGS; prediction-only Blue bonus V4=20, V3=0. No Elo updates after new matches. Not suitable for future seasons without a new lock.", "files_sha256": {p.name: sha(p) for p in sorted(SNAPS.iterdir()) if p.is_file()}}
        MANIFEST.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("New model and ratings snapshot locked. Push the frozen files and model_lock.json to GitHub before predicting.")
    info = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for name, expected_hash in info["files_sha256"].items():
        path = SNAPS / name
        if not path.exists() or sha(path) != expected_hash:
            raise RuntimeError(f"Frozen file changed: {name}. Restore it from Git history.")
    return info


def load_frozen_functions():
    """Import exactly the hashed code saved in forward_test/frozen/."""
    def load_module(name, filename):
        spec = importlib.util.spec_from_file_location(name, SNAPS / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    elo_module = load_module("frozen_forward_elo_model", "elo_model.py")
    series_module = load_module("frozen_forward_series_probability", "series_probability.py")
    return elo_module.elo_probability, series_module.series_win_probability


def select(prompt, choices):
    choices = list(choices)
    for i, name in enumerate(choices, 1):
        print(f"{i:>3}. {name}")
    while True:
        raw = input(prompt).strip()
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw)-1]
        print("Enter a listed number.")


def predict():
    model = select("Model: ", ["V3", "V4"])
    format_str = select("Match format: ", ["Bo1", "Bo3", "Bo5"])
    data = rows(SNAPS / RATINGS[model])
    if model == "V3":
        league = select("Domestic league: ", sorted({r["league"] for r in data}))
        data = [r for r in data if r["league"] == league]
        teams = sorted({r["team"] for r in data})
        a = select("Team A: ", teams)
        b = select("Team B: ", [t for t in teams if t != a])
        region_a = region_b = league
        elo_key = "elo_rating"
        bonus = 0
    else:
        # Allow any two different teams, including same-region knockouts.
        regions = sorted({r["region"] for r in data if r.get("year") == "2026"})
        region_a = select("Team A region: ", regions)
        region_b = select("Team B region (same region allowed): ", regions)
        subset_a = [r for r in data if r.get("year") == "2026" and r["region"] == region_a]
        subset_b = [r for r in data if r.get("year") == "2026" and r["region"] == region_b]
        a = select("Team A: ", sorted({r["team"] for r in subset_a}))
        b = select("Team B: ", sorted({r["team"] for r in subset_b if r["team"] != a}))
        bonus = 20
        elo_key = "elo"
    ra = next(r for r in data if r["team"] == a and (model == "V3" or (r["region"] == region_a and r["year"] == "2026")))
    rb = next(r for r in data if r["team"] == b and (model == "V3" or (r["region"] == region_b and r["year"] == "2026")))
    ea, eb = float(ra[elo_key]), float(rb[elo_key])
    # A Bo3/Bo5 uses neutral-side probability: future side assignments are unknown.
    elo_probability, series_win_probability = load_frozen_functions()
    p_game = elo_probability(ea, eb, bonus if format_str == "Bo1" else 0)
    p_match = p_game if format_str == "Bo1" else series_win_probability(p_game, int(format_str[-1]))
    print(f"\n{a} vs {b}: one-game {p_game:.2%}, {format_str} {p_match:.2%} for Team A")
    when = input("Scheduled match start UTC (YYYY-MM-DD HH:MM): ").strip()
    try:
        start = datetime.strptime(when, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError("Use UTC format YYYY-MM-DD HH:MM") from exc
    if start <= datetime.now(timezone.utc):
        raise ValueError("Match must be in the future; cannot log as a pre-match prediction")
    if input("Save this prediction? (yes/no): ").strip().lower() != "yes":
        print("Not saved.")
        return
    payload = dict(prediction_id=str(uuid.uuid4())[:12], recorded_at_utc=timestamp(), scheduled_at_utc=start.isoformat(), model=model, format=format_str, team_a=a, team_b=b, team_a_region=region_a, team_b_region=region_b, elo_a=ea, elo_b=eb, blue_bonus=(bonus if format_str == "Bo1" else 0), p_game_team_a=round(p_game, 10), p_series_team_a=round(p_match, 10), model_lock_sha256=sha(MANIFEST))
    if PREDS.exists():
        for existing in rows(PREDS):
            if all(existing[k] == str(payload[k]) for k in ("scheduled_at_utc", "model", "format", "team_a", "team_b")):
                raise ValueError("Duplicate prediction for same scheduled matchup")
    append(PREDS, PRED_FIELDS, payload)
    print(f"Saved ID {payload['prediction_id']} to {PREDS}. COMMIT & PUSH NOW before the game.")


def resolve():
    if not PREDS.exists():
        print("No predictions yet")
        return
    predictions = rows(PREDS)
    existing = {r["prediction_id"] for r in rows(RESULTS)} if RESULTS.exists() else set()
    pending = [r for r in predictions if r["prediction_id"] not in existing]
    if not pending:
        print("No pending predictions")
        return
    chosen = select("Prediction to resolve: ", [f"{r['prediction_id']}: {r['team_a']} vs {r['team_b']} ({r['format']})" for r in pending])
    pred = pending[[f"{r['prediction_id']}: {r['team_a']} vs {r['team_b']} ({r['format']})" for r in pending].index(chosen)]
    if datetime.fromisoformat(pred["scheduled_at_utc"]) > datetime.now(timezone.utc):
        raise ValueError("Scheduled start is in the future; do not resolve yet")
    winner = select("Actual winner: ", [pred["team_a"], pred["team_b"]])
    sa = input("Games won by Team A: ").strip()
    sb = input("Games won by Team B: ").strip()
    if not sa.isdigit() or not sb.isdigit():
        raise ValueError("Scores must be nonnegative integers")
    sa, sb = int(sa), int(sb)
    needed = int(pred["format"][-1]) // 2 + 1
    if (sa == needed) == (sb == needed) or max(sa,sb) != needed or min(sa,sb) >= needed or (winner == pred["team_a"]) != (sa > sb):
        raise ValueError("Final score and winner are inconsistent with match format")
    append(RESULTS, RESULT_FIELDS, dict(prediction_id=pred["prediction_id"], resolved_at_utc=timestamp(), winner=winner, score_a=sa, score_b=sb))
    print("Result appended to separate file. Keep predictions.csv unchanged. Commit & push results.csv.")


def report():
    if not PREDS.exists() or not RESULTS.exists():
        print("No completed forward predictions")
        return
    ps = {r["prediction_id"]: r for r in rows(PREDS)}
    finished = [(ps[r["prediction_id"]], r) for r in rows(RESULTS) if r["prediction_id"] in ps]
    if not finished:
        print("No completed forward predictions")
        return
    for group in ["ALL", "V3", "V4", "Bo1", "Bo3", "Bo5"]:
        matches = [(p,r) for p,r in finished if group == "ALL" or p["model"] == group or p["format"] == group]
        if not matches:
            continue
        probs = [min(1-1e-15, max(1e-15, float(p["p_series_team_a"]))) for p,r in matches]
        ys = [int(r["winner"] == p["team_a"]) for p,r in matches]
        accuracy = sum(int((pr >= .5) == bool(y)) for pr,y in zip(probs,ys))/len(matches)
        brier = sum((pr-y)**2 for pr,y in zip(probs,ys))/len(matches)
        logloss = -sum(y*math.log(pr)+(1-y)*math.log(1-pr) for pr,y in zip(probs,ys))/len(matches)
        print(f"{group}: n={len(matches)}, Accuracy={accuracy:.1%}, Brier={brier:.4f}, Log Loss={logloss:.4f} (50/50 baseline: 0.6931)")


if __name__ == "__main__":
    info = lock()
    print("\nFrozen baseline locked at", info["locked_at_utc"])
    choice = select("Action: ", ["Record a future prediction", "Record an actual result", "Report forward test metrics"])
    if choice.startswith("Record a future"):
        predict()
    elif choice.startswith("Record an actual"):
        resolve()
    else:
        report()
