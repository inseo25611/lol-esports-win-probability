
# League of Legends Esports Probability Model

**Elo-based pre-match probabilities for LCK/LEC and international matchups, with Bo1/Bo3/Bo5 series estimates, historical backtesting and calibration diagnostics.**

> Independent, non-commercial statistical portfolio demonstration. Not affiliated with Riot Games or Oracle's Elixir; no betting or real-money functionality. Ratings are historical snapshots, not live odds.

## What the application does

- **Domestic V3:** same-league LCK/LEC matchup estimates from league-specific Elo ratings (K=24, carryover=0.75).
- **Global V4:** global Elo ratings, including same-region pairings in the forward-test script; the historical V4 evaluation covers cross-region games only; Blue-side +20 is applied to single-game forecasts only, not to Elo updates.
- **Series forecast:** Bo3 and Bo5 win probabilities, plus exact final-score distributions computed from a single-game win probability. The series derivation assumes games have independent outcomes with the same win probability.
- **Historical Evaluation:** V1–V3 domestic and V4 international metrics, plus calibration plots showing mean forecast and observed win frequency and group sizes. Series scores use exploratory pre-series neutral-Elo predictions.
- **Quality assurance:** Pytest for Elo symmetry, side effects, series probabilities and CSV data quality; CI runs tests on push.

## Observed historical performance (2026)

| Dataset | Games | Accuracy | Brier ↓ | Log loss ↓ |
|---|---:|---:|---:|---:|
| Domestic V3 (LCK/LEC) | 885 | 63.50% | 0.2227 | 0.6355 |
| V4 (+20 Blue, cross-region only) | 221 | 60.63% | 0.2272 | 0.6450 |

Evaluation sets differ; do not directly compare these as a head-to-head model ranking. Model development involved inspecting 2025–2026 results; they are **exploratory chronological backtests**, not untouched holdouts. **2024 is a warm-up season:** ratings begin near 1500, so its accuracy should not be interpreted as a clean prospective test.

### Exploratory series sample

The selected sample has 33 series (10 Bo3, 23 Bo5), mostly from 2024. `data/series_source_audit.csv` provides reference links, but individual results have not been documented as independently checked. The app compares aggregate log loss (0.6489) with a constant-50% baseline (0.6931), with format-specific drill-down; this small selected sample is exploratory.

Pre-series prediction uses first-game pre-match Elo with **no +20 side bonus**, because later game side assignments are not reliably known before the series. Future-side-aware calculations would be retrospective diagnostics only, not pre-series forecasts.

## Model selection

The existing V3 and V4 models are retained rather than re-tuned on the reported evaluation results. During development, K-factor, seasonal carryover, Blue-side adjustment, and recent-form and logistic regression alternatives.

The repository includes reproducible notebooks for the domestic and international Elo models, alongside evaluation scripts and recorded prediction data. Historical results are treated as exploratory backtests rather than untouched holdout performance.

Model selection considers probabilistic accuracy and calibration, including Brier score and log loss, rather than classification accuracy alone.

## Data and provenance

Source: [Oracle's Elixir](https://master.d36liwrx5rvjnc.amplifyapp.com/tools/downloads), Tim Sevenhuysen. Original annual CSV files are not redistributed. V3 covers 2,524 LCK/LEC single games in 2024–2026. V4 ingests 9,863 selected games; evaluation excludes same-region matchups. `WLDs` includes LPL regional qualification rounds; **99 games recorded under `WLDs` are same-region** and are not counted in cross-region evaluation.

Historical team records are retained even when the demo hides academy or inactive teams. Name normalization and historical regional affiliation should not be confused with guaranteed current professional membership.

## Reproducibility

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest -q tests/
python scripts/build_domestic_evaluation.py
python scripts/build_series_evaluation.py
python -m streamlit run app.py
```

`notebooks/lol_elo_clean.ipynb` and `notebooks/lol_international_elo_v4.ipynb` contain the chronological Elo experiments. The V4 notebook includes a single match-level CSV export cell; running full data reconstruction requires the original Oracle's Elixir source files. `scripts/build_domestic_evaluation.py` reconstructs exported domestic metric and calibration tables from recorded pre-game probabilities. `scripts/build_series_evaluation.py` reconstructs the exploratory selected-series table from recorded source audit and pre-game Elo, without automatically promoting any entry to “verified”.

## Repository structure

- `app.py`, `elo_model.py`, `series_probability.py`: app and prediction functions
- `data/`: rating snapshots, per-game probabilities, metrics, calibration, series audit and exploratory results
- `scripts/`: evaluation CSV generation
- `notebooks/`: Elo analyses and V4 match-level export
- `tests/`: model and data-quality tests
- `.github/workflows/test.yml`: automated pytest
- `forward_test.py`, `forward_test/`: prospective pre-match predictions, locked model code and ratings, SHA-256 integrity checks and separately recorded results

## Forward test (in progress)

Model code and Elo ratings were locked on **2 November 2026 at 16:07 UTC**, with SHA-256 hashes recorded in `forward_test/model_lock.json`. The forward-test script imports the locked code and uses the locked rating snapshot. Predictions are recorded before their respective match results are known, and results are stored separately.

| Match | Format | Team A predicted series win probability |
|---|---|---:|
| Top Esports vs Invictus Gaming | Bo5 | 49.77% |
| Gen.G vs LYON | Bo5 | 84.62% |

`forward_test/predictions.csv` contains the two recorded predictions. GitHub's remote commit history can be used to check whether each version of the prediction file was published before the corresponding match began. The local lock timestamp by itself does not establish when a prediction was published.

**Scope:** This is a two-match prospective demonstration of the workflow, not a statistically meaningful performance evaluation. The ratings remain fixed across matches, so subsequent tournament results are not reflected. The lock date refers to when this snapshot was frozen, **not** a separately verified source-data cutoff date. Do not reuse the snapshot for a future season without updating ratings and applying seasonal carryover.

## Limitations

Elo does not explicitly model draft, patches, roster changes or intra-series dependence; win probabilities may be miscalibrated across regions. Selected series sample is small and cannot be used to claim reliable Bo3/Bo5 generalisation. The application offers historical Elo-based estimates, while the separate forward-test script records pre-match predictions. Neither provides live in-game predictions, bookmaker odds, betting recommendations or profitability analysis.

## Key modelling choices

**Bo3 formula:** Win 2–0 has probability `p²`, win 2–1 has two valid game sequences, each `p²(1-p)`, so `P(win Bo3)=p²(3−2p)`.

**Why independence may fail:** Drafting, adaptation, changing sides, patches and psychology can affect game-to-game win chances.

**Why no side bonus for the series backtest:** At series start, not all future game sides are guaranteed to be known. Applying the first game's Blue bonus to all games would introduce an unjustified assumption; using realised future sides would leak information.
