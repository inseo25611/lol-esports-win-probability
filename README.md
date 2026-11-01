# League of Legends Esports Win Probability Prediction

**An Elo-based probabilistic forecasting project for professional League of Legends, covering domestic and international matchups (2024–2026).**

This project estimates pre-match win probabilities using chronological match results from [Oracle's Elixir](https://master.d36liwrx5rvjnc.amplifyapp.com/tools/downloads). It compares model variants, measures the quality of predicted probabilities, and presents estimates in a two-mode Streamlit application.

> **Evaluation note:** Results below are exploratory backtests, not independent prospective forecasts. The 2025 and 2026 results were inspected during development. No bookmaker-odds comparison or betting-profit claim is made.

## At a glance

| | V3 — Domestic Leagues | V4 — International Matchups |
|---|---|---|
| Purpose | Predict matchups within LCK or LEC | Estimate matchups across regional leagues |
| Rating structure | Independent Elo scale for each league | Shared team-level Elo scale connected by international results |
| Model input | 2,524 LCK/LEC games | 9,863 selected domestic and international games |
| 2026 evaluation | 885 LCK/LEC games | 221 cross-region games |
| Accuracy | **63.50%** | **60.63%** |
| Log Loss ↓ | **0.6355** | **0.6450** |
| Brier Score ↓ | **0.2227** | **0.2272** |
| Blue-side adjustment | None | +20 Elo applied to displayed probabilities only |

**Do not directly rank V3 against V4 using these scores:** their evaluation populations differ.

## Interactive application

The Streamlit interface has two modes:

- **Domestic Leagues (V3):** choose LCK or LEC and two teams within that league. View estimated win probabilities, Elo ratings and a visual comparison.
- **International Matchups (V4):** choose teams from different regional groups, designate Blue and Red sides, and compare global Elo estimates and win probabilities visually. The probability calculation adds a fixed **+20 Elo** to the Blue team's rating difference.

The app is designed as an explanatory portfolio demo, not a live match predictor. Ratings are historical snapshots, not automatically refreshed. Its region labels are analysis categories rather than guarantees of current league membership.

### Run locally

Place `app.py`, `requirements.txt`, `latest_team_ratings.csv`, and `international_team_ratings.csv` in the same folder. Then run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL shown in the terminal (usually `http://localhost:8501`).

## Data and preparation

**Source:** [Oracle's Elixir](https://master.d36liwrx5rvjnc.amplifyapp.com/tools/downloads) (Tim Sevenhuysen)  
**Period:** 2024–2026  
**Original CSVs:** Not included in this repository.

I combined annual match files, retained team-level records, checked match IDs and outcomes, reconstructed Blue/Red game-level observations, and ordered matches chronologically. Predictions are recorded **before** the corresponding result updates the Elo ratings.

### V3 data: LCK and LEC

| Year | LCK | LEC | Total |
|---|---:|---:|---:|
| 2024 | 482 | 294 | 776 |
| 2025 | 555 | 308 | 863 |
| 2026 | 502 | 383 | 885 |
| **Total** | **1,539** | **985** | **2,524** |

One inconsistent game in the source preparation process was removed. Team names were normalised for selected rebrandings while preserving their original forms.

### V4 data: domestic leagues and international competitions

The V4 dataset contains **9,863 unique games** and **186 team identities** after selected name mappings:

| Type | Included games |
|---|---:|
| Domestic regional competitions | 9,138 |
| MSI, Worlds (`WLDs`) and First Stand (`FST`) | 622 |
| Esports World Cup (EWC) main-event candidates | 103 |
| **Total** | **9,863** |

Domestic competition codes include LCK, LPL, LEC, LCS, LTA, LTA N, LTA S, CBLOL, PCS, VCS, LJL, LCP and LLA. EWC selection retains the 2024 and 2025 recorded matches and the July 2026 event; the April–June 2026 records are excluded because they appear to include qualifiers rather than the main international event. This is a documented selection rule, not independent verification of every fixture.

For evaluation, domestic league affiliations were mapped to broad groups (Korea, China, Europe, North America, South America, Latin America and Asia Pacific). Mixed-region LTA records were not used to assign a team's primary region; the most represented specific domestic league within the year was used instead. The categories simplify some distinctions, especially within Asia Pacific.

Selected canonical team names include `MAD Lions KOI` → `Movistar KOI` and `Team BDS` → `Shifters`. The **2026 EWC** `Team Secret` records were mapped to `Team Secret Whales`, while **2024 VCS** `Team Secret` remains a separate historical identity. `Movistar R7` and `LYON` were not automatically merged. These are explicit identity assumptions, not claims of unchanged rosters.

## Methodology

### Elo predictions and updates

Initial rating: **1500**. K-factor: **24**. Seasonal carryover: **0.75**.

For a Blue-side team with rating \(R_B\) and a Red-side team with rating \(R_R\), the base probability is:

\[
P(\text{Blue wins}) = \frac{1}{1+10^{(R_R-R_B)/400}}
\]

After the match, each team is updated by:

\[
R_{\mathrm{new}} = R_{\mathrm{old}} + K(S-P)
\]

Here \(S\) is the observed outcome (1 for a win, 0 for a loss), and \(P\) is the team's corresponding pre-match win probability. At a team's first match in a new calendar year, the rating is regressed toward 1500:

\[
R_{\mathrm{next\ year}} = 1500 + 0.75(R_{\mathrm{previous}}-1500)
\]

**V3:** team ratings are keyed by `(league, team)`; LCK and LEC Elo scales are independent and **cannot** be compared directly for cross-league matchups. No Blue-side bonus is used.

**V4:** team ratings are keyed by canonical team identity across domestic and international competitions. International matches allow results to propagate across regions, although some teams and regions are connected by relatively few cross-region games.

For the V4 app's **displayed** Blue-side probability only:

\[
P_{\mathrm{adjusted}}(\text{Blue wins}) =
\frac{1}{1+10^{-(R_B-R_R+20)/400}}
\]

The +20 Elo is a **post-hoc prediction adjustment**; the underlying Elo ratings were still updated using the unadjusted probability. The app must therefore not be described as a fully re-trained Blue-side Elo system.

### V3 model experiments

I evaluated K-factors of 8, 16, 24, 32, 48, 64 and 96, selected K=24 using 2025 probabilistic metrics, and tested canonical name mapping, seasonal regression, Blue-side adjustments, recent-five-game win rates and Logistic Regression features. The final domestic model retains team-name mapping and 75% seasonal carryover, without a Blue-side bonus.

| Model | 2026 accuracy | 2026 Log Loss ↓ | 2026 Brier ↓ |
|---|---:|---:|---:|
| V1 — basic Elo | 63.05% | 0.6395 | 0.2242 |
| V2 — unified names | 63.39% | 0.6388 | 0.2237 |
| **V3 — seasonal Elo** | **63.50%** | **0.6355** | **0.2227** |

2025 V3 validation: 66.98% accuracy, 0.6132 Log Loss, 0.2122 Brier Score. In 2026, LCK achieved 63.15% accuracy (502 games) and LEC 63.97% (383 games). Individual high-loss upsets and team-level observed-versus-predicted win rates were also examined.

### V4 cross-region evaluation

V4 ratings were updated chronologically across selected domestic and international games. The cross-region subset was identified by teams' year-specific domestic affiliations.

| Year | Cross-region games | Base V4 accuracy | Base V4 Log Loss ↓ | Base V4 Brier ↓ |
|---|---:|---:|---:|---:|
| 2024 | 183 | 50.82% | 0.7047 | 0.2546 |
| 2025 | 196 | 59.69% | 0.6721 | 0.2396 |
| 2026, before final team-alias correction | 217 | 59.45% | 0.6518 | 0.2304 |

These intermediate figures describe the exploration before the final correction, and should **not** be confused with the corrected 2026 evaluation below.

Using 2024–2025 cross-region matches, I compared candidate fixed Blue-side adjustments of −40, −20, 0, +20, +40 and +60 Elo. **+20** had the lowest training-period Log Loss among those options. I then evaluated it on the already-examined 2026 season. This was **not** an untouched blind test.

After correcting the 2026 EWC Team Secret identity, all **221** 2026 cross-region games had assigned regions. The final exploratory comparison was:

| Model | Accuracy ↑ | Log Loss ↓ | Brier ↓ |
|---|---:|---:|---:|
| Pre-correction V4 predictions on the same 221 games | 59.28% | 0.6527 | 0.2308 |
| Corrected V4, no Blue adjustment | 59.28% | 0.6548 | 0.2318 |
| **Corrected V4 +20 Blue adjustment** | **60.63%** | **0.6450** | **0.2272** |
| Constant 50% probability baseline | — | 0.6931 | 0.2500 |

Compared with the constant-probability baseline, the adjusted 2026 V4 has approximately **6.9% lower Log Loss** and **9.1% lower Brier Score**. These results concern game-level probability predictions, not betting returns.

Calibration was assessed in five equal-frequency bins. The +20 adjustment moved the 2026 bin averages closer to observed Blue win rates in this exploratory dataset, but differences and sample sizes do not establish general calibration on new competitions.

## Repository files

| File | Description |
|---|---|
| `lol_elo_clean.ipynb` | V1–V3 domestic modelling and evaluation |
| `lol_international_elo_v4.ipynb` | V4 data checks, global Elo experiments, cross-region evaluation and export |
| `lol_elo_model_comparison.csv` | Domestic model comparison |
| `latest_team_ratings.csv` | Domestic V3 rating snapshot |
| `international_team_ratings.csv` | Global V4 rating snapshot and region labels |
| `app.py` | Streamlit interface with Domestic and International modes |
| `requirements.txt` | App dependencies |

### Reproduce the notebooks

Obtain the 2024, 2025 and 2026 annual match datasets directly from Oracle's Elixir and make them available to the notebooks under the expected file names, such as `2024_LoL_esports_match_data_from_OraclesElixir.csv`. Open the notebooks in Google Colab or Jupyter, install required packages (including pandas and scikit-learn), and execute cells in order. The original provider CSV files are not redistributed here.

**Tools:** Python, pandas, scikit-learn, Google Colab/Jupyter and Streamlit.

## Limitations and responsible use

- **Exploratory evaluation:** 2025 and 2026 results were examined repeatedly during development; reported values are not unbiased estimates of future performance.
- **Different evaluation sets:** V3 and V4 results cover different leagues and match populations, so cross-version comparisons are not head-to-head performance tests.
- **Incomplete information:** Elo does not model drafts, champions, patch changes, side selection mechanisms, player availability, roster changes or all tournament-format effects.
- **Regional uncertainty:** Global Elo is only weakly linked across some regions; a high rating from a region with limited international exposure is not necessarily a reliable global ranking.
- **Team identity:** Name mapping simplifies organisational change and does not preserve roster composition. The latest region shown is historical metadata, not a guarantee of current eligibility or affiliation.
- **Rating snapshot:** The CSVs reflect the recorded historical period and are not live-updated. V3's reported analysis ends in September 2026; V4 includes selected events recorded later in 2026.
- **No market validation:** No bookmaker prices, edge estimates, expected value, transaction costs or profitability are analysed. The model is not intended as betting advice.

## Data attribution

Match data: [Oracle's Elixir](https://master.d36liwrx5rvjnc.amplifyapp.com/tools/downloads), created by Tim Sevenhuysen. Original datasets are not redistributed. Any reuse or publication of derived data and demonstrations should comply with the provider's terms and applicable Riot Games policies; attribution alone does not establish permission.

This is an independent, non-commercial educational portfolio project, not affiliated with or endorsed by Oracle's Elixir, Riot Games or participating esports organisations.
