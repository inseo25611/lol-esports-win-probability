"""One-file update for LoL project: run inside the GitHub repository folder.

Usage (Windows CMD): python apply_lol_cleanup.py
Creates backups, updates app.py, removes one duplicated V4 Colab export cell,
and writes tests/test_data_quality.py. Keeps source CSVs and all Elo formulas unchanged.
"""
from pathlib import Path
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parent
APP = ROOT / 'app.py'
NB = ROOT / 'lol_international_elo_v4.ipynb'
TEST = ROOT / 'tests' / 'test_data_quality.py'

# Explicit, intentionally conservative candidate lists based on supplied 2026
# rating snapshot. NOT a live roster feed or official current roster certification.
FIRST_TEAMS = {
    'Korea': ['Gen.G', 'Hanwha Life Esports', 'Dplus Kia', 'T1', 'KT Rolster', 'BNK FEARX',
              'HANJIN BRION', 'Nongshim RedForce', 'DN SOOPers', 'Kiwoom DRX'],
    'China': ['Bilibili Gaming', 'Invictus Gaming', 'Top Esports', 'Ninjas in Pyjamas',
              "Anyone's Legend", 'JD Gaming', 'LGD Gaming', 'Team WE', 'Weibo Gaming',
              'LNG Esports', 'EDward Gaming', 'ThunderTalk Gaming', 'Ultra Prime', 'Oh My God'],
    'Europe': ['Karmine Corp', 'G2 Esports', 'Movistar KOI', 'Team Vitality', 'GiantX',
               'Natus Vincere', 'Fnatic', 'SK Gaming', 'Shifters', 'Team Heretics'],
    'North America': ['LYON', 'Team Liquid', 'Cloud9', 'FlyQuest',
                      'Shopify Rebellion', 'Dignitas'],
    'South America': ['LØS', 'FURIA', 'LOUD', 'RED Canids', 'Vivo Keyd Stars',
                      'paiN Gaming', 'Fluxo W7M'],
    'Asia Pacific': ['CTBC Flying Oyster', 'Frank Esports', 'GAM Esports',
                     'MVK Esports', 'Team Secret Whales', 'Fukuoka SoftBank HAWKS gaming',
                     'DetonatioN FocusMe', 'Ground Zero Gaming'],
}

APP_HELPERS = '''\n\n# Snapshot-based first-team selection only; this is NOT a live 2026 roster source.
FIRST_TEAM_SELECTION = __FIRST_TEAM_SELECTION__


def filter_first_team_ratings(df):
    """Whitelist named first-team candidates; preserve underlying source data."""
    eligible = {(region, team) for region, teams in FIRST_TEAM_SELECTION.items()
                for team in teams}
    return df.loc[[ (region, team) in eligible and int(year) == 2026
                    for region, team, year in
                    zip(df["region"], df["team"], df["year"]) ]].copy()


def validate_ratings(df, *, international=False):
    import numpy as np
    key = ["team", "region"] if international else ["league", "team"]
    rating = "elo" if international else "elo_rating"
    required = set(key + [rating] + (["year"] if international else []))
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Rating CSV is missing columns: {sorted(missing)}")
    if df[key].isna().any().any() or df[key].duplicated().any():
        raise ValueError("Missing or duplicated team identifiers in rating CSV")
    ratings = pd.to_numeric(df[rating], errors="coerce")
    if not np.isfinite(ratings.to_numpy()).all():
        raise ValueError("Rating CSV contains invalid or non-finite Elo")
    if international and pd.to_numeric(df["year"], errors="coerce").isna().any():
        raise ValueError("International rating CSV has invalid years")
    return df


def validate_match_predictions(df, *, international=False):
    import numpy as np
    required = {"gameid", "date", "blue_win"}
    probability_cols = (["v4_blue_prob", "v4_adjusted_prob"] if international
                        else ["v1_blue_prob", "v2_blue_prob", "v3_blue_prob"])
    required.update(probability_cols)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Match CSV is missing columns: {sorted(missing)}")
    if df["gameid"].isna().any() or df["gameid"].duplicated().any():
        raise ValueError("Match CSV contains empty/duplicate game IDs")
    if pd.to_datetime(df["date"], errors="coerce").isna().any():
        raise ValueError("Match CSV contains unparseable dates")
    result = pd.to_numeric(df["blue_win"], errors="coerce")
    if not result.isin([0, 1]).all():
        raise ValueError("blue_win must be 0 or 1")
    for col in probability_cols:
        p = pd.to_numeric(df[col], errors="coerce").to_numpy()
        if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
            raise ValueError(f"Invalid probability values in {col}")
    return df
'''.replace('__FIRST_TEAM_SELECTION__', repr(FIRST_TEAMS))

TEST_CONTENT = '''import pathlib
import sys
import pandas as pd
import pytest

# Keep data checks independent of Streamlit rendering.
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# app.py is intentionally not imported because it executes Streamlit UI at import.
# This test checks real CSV invariants and the published first-team selection.

@pytest.mark.parametrize("filename,required", [
    ("latest_team_ratings.csv", {"league", "team", "elo_rating"}),
    ("international_team_ratings.csv", {"team", "region", "year", "elo"}),
    ("lol_match_predictions_final.csv", {"gameid", "date", "blue_win", "v3_blue_prob"}),
    ("v4_match_predictions_final.csv", {"gameid", "date", "blue_win", "v4_adjusted_prob"}),
])
def test_csv_required_fields(filename, required):
    df = pd.read_csv(ROOT / filename)
    assert required.issubset(df.columns)

@pytest.mark.parametrize("filename", [
    "lol_match_predictions_final.csv", "v4_match_predictions_final.csv"
])
def test_predictions_valid(filename):
    df = pd.read_csv(ROOT / filename)
    assert df.gameid.notna().all()
    assert not df.gameid.duplicated().any()
    assert pd.to_datetime(df.date, errors="coerce").notna().all()
    assert pd.to_numeric(df.blue_win, errors="coerce").isin([0, 1]).all()
    prob_cols = [c for c in df.columns if c.endswith("_prob")]
    assert prob_cols
    for col in prob_cols:
        assert pd.to_numeric(df[col], errors="coerce").between(0, 1).all(), col

@pytest.mark.parametrize("filename,cols,rating", [
    ("latest_team_ratings.csv", ["league", "team"], "elo_rating"),
    ("international_team_ratings.csv", ["region", "team"], "elo"),
])
def test_ratings_valid(filename, cols, rating):
    import numpy as np
    df = pd.read_csv(ROOT / filename)
    assert not df[cols].isna().any().any()
    assert not df[cols].duplicated().any()
    assert np.isfinite(pd.to_numeric(df[rating], errors="coerce").to_numpy()).all()


def test_academy_absent_from_app_allowlist():
    import ast
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    assignments = [n for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "FIRST_TEAM_SELECTION"
                           for t in n.targets)]
    assert len(assignments) == 1
    allowlist = ast.literal_eval(assignments[0].value)
    names = [name for teams in allowlist.values() for name in teams]
    assert len(names) == len(set(names))
    assert all(not any(word in n.casefold() for word in
                       ("academy", "youth", "karmine corp blue")) for n in names)
    df = pd.read_csv(ROOT / "international_team_ratings.csv")
    missing = [(region, name) for region, teams in allowlist.items() for name in teams
               if not ((df.region == region) & (df.team == name) & (df.year == 2026)).any()]
    assert not missing, f"Allowlisted team missing from 2026 snapshot: {missing}"
'''

def patch_app(source):
    if 'FIRST_TEAM_SELECTION =' in source:
        raise RuntimeError('app.py already updated. No second patch applied.')
    needle = 'ACADEMY_TEAMS = {"Karmine Corp Blue"}'
    if needle not in source:
        raise RuntimeError('Expected academy list not found; app.py differs from inspected version')
    source = source.replace(needle, needle + APP_HELPERS, 1)
    source = source.replace('return pd.read_csv("latest_team_ratings.csv")',
                            'return validate_ratings(pd.read_csv("latest_team_ratings.csv"))', 1)
    source = source.replace('return pd.read_csv("international_team_ratings.csv")',
                            'return validate_ratings(pd.read_csv("international_team_ratings.csv"), international=True)', 1)
    source = source.replace('        return pd.read_csv(path)',
                            '        df = pd.read_csv(path)\n'
                            '        if path == "lol_match_predictions_final.csv":\n'
                            '            validate_match_predictions(df)\n'
                            '        elif path == "v4_match_predictions_final.csv":\n'
                            '            validate_match_predictions(df, international=True)\n'
                            '        return df', 1)
    source = source.replace('    df = df[~df["team"].isin(ACADEMY_TEAMS)].copy()\n'
                            '    st.caption("Known academy team excluded: Karmine Corp Blue. Other regional teams require a separate roster audit; this is not a full current-season filter.")',
                            '    original_count = len(df)\n'
                            '    df = filter_first_team_ratings(df)\n'
                            '    st.caption(f"Showing {len(df)} conservatively selected first-team candidates "\n'
                            '               f"from the 2026 rating snapshot (out of {original_count} historical entries). "\n'
                            '               "This is not a live roster list; source CSV and historical evaluations are unchanged.")', 1)
    if 'original_count = len(df)' not in source:
        raise RuntimeError('Unable to locate V4 filter block')
    source = source.replace('Series outcomes have not been separately backtested.',
                            'Limited exploratory series backtesting is available in Historical Model Evaluation; it is not an independently held-out evaluation.', 1)
    return source


def patch_notebook(doc):
    cells = doc.get('cells', [])
    export_indices = [i for i, c in enumerate(cells)
                      if 'v4_match_predictions_final.csv' in ''.join(c.get('source', []))
                      and 'Paste this into ONE new cell' in ''.join(c.get('source', []))]
    if len(export_indices) >= 2:
        kept = export_indices[0]
        for ix in reversed(export_indices[1:]):
            cells.pop(ix)
        print(f'Removed {len(export_indices)-1} duplicate V4 export cell(s); retained cell {kept}.')
    else:
        print('No duplicated export cell detected; notebook unchanged.')
    return doc


def main():
    if not APP.is_file() or not NB.is_file():
        sys.exit('ERROR: Place this script beside app.py and lol_international_elo_v4.ipynb (repository root).')
    original = APP.read_text(encoding='utf-8')
    modified = patch_app(original)
    compile(modified, str(APP), 'exec')
    notebook = json.loads(NB.read_text(encoding='utf-8'))
    notebook = patch_notebook(notebook)
    for f in [APP, NB]:
        backup = f.with_name(f.name + '.bak')
        if not backup.exists():
            shutil.copy2(f, backup)
    APP.write_text(modified, encoding='utf-8')
    NB.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    TEST.parent.mkdir(parents=True, exist_ok=True)
    TEST.write_text(TEST_CONTENT, encoding='utf-8')
    print('Updated: app.py, lol_international_elo_v4.ipynb, tests/test_data_quality.py')
    print('Backups: app.py.bak, lol_international_elo_v4.ipynb.bak (do not upload backups to GitHub)')
    print('Run: python -m pytest -q tests/')

if __name__ == '__main__':
    main()
