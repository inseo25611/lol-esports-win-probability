# lol-esports-win-probability# League of Legends Esports Win Probability Prediction

**An Elo-based probabilistic modelling project using professional LCK and LEC match data (2024–2026).**

## Project Overview

How accurately can we estimate the probability of one professional League of Legends team defeating another using historical match results?

This project develops and evaluates an Elo-based win probability model for two professional esports leagues:

- **LCK:** League of Legends Champions Korea
- **LEC:** League of Legends EMEA Championship

Using match data from Oracle's Elixir, I built a chronological rating system that updates team strength after each match and estimates win probabilities using information available before the match result.

The project focuses on probabilistic forecasting, model evaluation and the trade-off between model complexity and predictive performance.

## Key Results

The final experimental model (V3) achieved the following results across **885 LCK and LEC games in 2026**:

| Metric | Result |
|---|---:|
| Accuracy | **63.50%** |
| Log Loss | **0.6355** |
| Brier Score | **0.2227** |

These results are exploratory rather than a fully independent test. Both 2025 and 2026 results were examined during model development, so they should not be interpreted as unbiased estimates of future predictive performance.

## Data

**Source:** [Oracle's Elixir — League of Legends Esports Data](https://master.d36liwrx5rvjnc.amplifyapp.com/tools/downloads)

**Period:** 2024–2026

**Leagues:** LCK and LEC

**Final dataset:** 2,524 individual games

| Season | LCK | LEC | Total |
|---|---:|---:|---:|
| 2024 | 482 | 294 | 776 |
| 2025 | 555 | 308 | 863 |
| 2026 | 502 | 383 | 885 |
| **Total** | **1,539** | **985** | **2,524** |

### Data Preparation

The source data contains player-level and team-level records, with multiple rows representing each game.

I prepared the modelling dataset by:

1. Combining annual source files and isolating team-level records.
2. Validating match identifiers, team records and winning outcomes.
3. Excluding one game with inconsistent winning-team information.
4. Restructuring the data into one row per game, containing Blue and Red team information.
5. Filtering for LCK and LEC matches and sorting them chronologically.
6. Preserving historical team names while creating consistent team identifiers for Elo modelling.

The original source datasets are not redistributed in this repository.

## Methodology

### 1. Elo Rating System

Elo is a rating method that estimates relative team strength from historical results.

Each team begins with a rating of 1500. Before each match, the model calculates the probability that the Blue-side team wins:

\[
P(\text{Blue wins}) =
\frac{1}{1 + 10^{(R_{\text{Red}} - R_{\text{Blue}})/400}}
\]

where \(R_{\text{Blue}}\) and \(R_{\text{Red}}\) are the teams' pre-match Elo ratings.

After each game, ratings are updated using:

\[
R_{\text{new}} = R_{\text{old}} + K(S-P)
\]

where:

- \(K\) controls the magnitude of rating updates.
- \(S\) is the actual match outcome (1 for a win, 0 for a loss).
- \(P\) is the predicted win probability.

**All predictions are recorded before the corresponding game result is used to update ratings.**

LCK and LEC ratings are maintained separately. The model does not directly support cross-league matchups.

### 2. Selecting the K-Factor

I compared K-factors of 8, 16, 24, 32, 48, 64 and 96 using 2025 validation results.

K=24 produced the lowest Log Loss and Brier Score among the tested values and was selected for subsequent experiments.

The objective was to estimate reliable probabilities rather than maximise classification accuracy alone.

### 3. Additional Feature Experiments

I investigated whether other information improved predictive performance.

**Blue-side adjustment**

I evaluated a fixed Elo bonus for teams playing on the Blue side. This improved classification accuracy in some comparisons but did not consistently improve probabilistic performance across seasons.

The final Elo model therefore uses no fixed Blue-side bonus.

**Recent match form**

I calculated each team's win rate over its previous five games, excluding the current match to prevent target leakage.

I also compared Logistic Regression models using:

- Pre-match Elo difference only
- Pre-match Elo difference and recent five-game win-rate difference

The additional feature did not consistently improve probabilistic predictions over the original Elo approach.

These experiments helped assess whether extra features provided useful information beyond the existing team ratings.

### 4. Team Identity Mapping

Team rebranding can cause rating discontinuities when the same team appears under different names.

I tested the following identity mappings:

| Historical name | Canonical modelling name |
|---|---|
| MAD Lions KOI | Movistar KOI |
| Team BDS | Shifters |

Original names are preserved separately from the canonical identifiers.

Rogue and Natus Vincere are treated as separate identities rather than automatically transferring their Elo ratings.

These mappings are modelling assumptions about organisational continuity; they do not imply an unchanged roster or team strength.

### 5. Seasonal Elo Regression

Professional esports teams may change rosters, coaching staff and strategies between seasons.

To reflect the possibility that past performance becomes less informative over time, I tested seasonal rating regression towards the initial Elo of 1500:

\[
R_{\text{new season}} = 1500 + \alpha(R_{\text{previous}} - 1500)
\]

I compared several values of \(\alpha\) using 2025 validation data.

A carryover factor of **0.75** was selected, preserving 75% of each team's rating deviation from 1500 when it first plays in a new calendar year.

This adjustment produced a modest improvement in the exploratory 2026 evaluation.

## Model Comparison

Three Elo configurations were reconstructed and compared:

| Model | Description |
|---|---|
| **V1 — Basic Elo** | K=24, historical team names treated separately, no seasonal regression |
| **V2 — Unified Team Names** | V1 with canonical team identity mapping |
| **V3 — Seasonal Elo** | V2 with 75% seasonal Elo carryover |

### 2025 Validation Results

| Model | Accuracy | Log Loss ↓ | Brier Score ↓ |
|---|---:|---:|---:|
| V1 | 67.21% | 0.6139 | 0.2121 |
| V2 | 67.21% | 0.6140 | 0.2122 |
| V3 | 66.98% | **0.6132** | 0.2122 |

### 2026 Exploratory Evaluation

| Model | Accuracy ↑ | Log Loss ↓ | Brier Score ↓ |
|---|---:|---:|---:|
| V1 | 63.05% | 0.6395 | 0.2242 |
| V2 | 63.39% | 0.6388 | 0.2237 |
| **V3** | **63.50%** | **0.6355** | **0.2227** |

V3 achieved the lowest Log Loss and Brier Score among the three Elo configurations in the 2026 exploratory comparison.

However, the differences are small. These results do not establish that V3 will consistently outperform the alternatives on future unseen data.

## Model Evaluation

Three principal metrics were used:

**Accuracy:** The proportion of games for which the model correctly predicted the winning side using a 50% probability threshold.

**Log Loss:** A probabilistic scoring rule that places a larger penalty on confident incorrect predictions.

**Brier Score:** The mean squared difference between predicted win probabilities and actual binary outcomes.

I also examined probability calibration, individual high-loss matches, and team-level differences between predicted and observed win rates.

### League-Level Results

V3 performance in 2026:

| League | Matches | Accuracy | Log Loss | Brier Score |
|---|---:|---:|---:|---:|
| LCK | 502 | 63.15% | 0.6374 | 0.2238 |
| LEC | 383 | 63.97% | 0.6330 | 0.2213 |

The differences between leagues are relatively small and do not establish that one league is inherently more predictable.

### Error Analysis

I examined games where the model assigned a very low probability to the actual winner.

For example, the model assigned Shifters a 9.4% probability of defeating Karmine Corp in one 2026 match that Shifters ultimately won.

Such outcomes receive high Log Loss penalties, but individual upsets do not necessarily indicate that a probabilistic model is incorrectly specified.

I also compared team-level predicted and observed win rates to identify potential systematic differences and distinguish isolated upsets from broader prediction patterns.

## Reproducibility

### Repository Files

| File | Purpose |
|---|---|
| `lol_elo_clean.ipynb` | Reproducible data preparation, Elo modelling and evaluation workflow |
| `lol_elo_model_comparison.csv` | Summary of V1, V2 and V3 performance |

### Running the Analysis

1. Download the 2024, 2025 and 2026 annual match datasets from [Oracle's Elixir](https://master.d36liwrx5rvjnc.amplifyapp.com/tools/downloads).
2. Place the files in the same directory as the notebook.
3. Rename the annual files to the names expected by the notebook, following this pattern:

   `2024_LoL_esports_match_data_from_OraclesElixir.csv`

   Repeat for 2025 and 2026.

4. Open `lol_elo_clean.ipynb` in Jupyter Notebook or Google Colab, ensuring the source CSVs are accessible from the working directory.
5. Install the required Python packages if needed:

   `pip install pandas scikit-learn`

6. Execute the notebook sequentially.

The notebook generates model comparisons, match-level prediction outputs and a team-rating snapshot for further analysis.

### Technologies

Python · pandas · scikit-learn · Jupyter Notebook / Google Colab · Elo Rating System

## Limitations

This project has several limitations:

- **Evaluation reuse:** 2025 was used repeatedly for model development, and 2026 results were inspected during experimentation. Neither should be presented as a fully untouched final holdout.
- **Simplified team strength:** Elo uses historical match outcomes but does not explicitly model champion selection, player-level performance, injuries, roster changes or patch-specific effects.
- **Team continuity:** Canonical team mapping reflects selected organisational changes but does not fully represent changes in player composition.
- **Season boundaries:** Seasonal regression is applied at calendar-year transitions rather than at every roster or competitive split change.
- **Cross-league comparisons:** LCK and LEC have independent Elo rating scales, so predictions between teams from different leagues are not currently supported.
- **Data coverage:** The analysed dataset ends in September 2026. Subsequent matches are not reflected in the reported ratings.
- **No betting-market validation:** The project does not use bookmaker odds, estimate betting-market value, or demonstrate financial returns.

## Next Steps

The next phase is to develop an interactive Streamlit application where users can:

- Select LCK or LEC.
- Choose two teams from the selected league.
- View each team's Elo rating.
- Calculate the estimated win probability for a hypothetical matchup.
- See the date of the latest available rating update.

The application will use the existing V3 model framework and clearly distinguish model estimates from actual match outcomes.

## Data Attribution & Usage

This is an independent, non-commercial portfolio project developed for educational and demonstration purposes.

Match data was sourced from [Oracle's Elixir](https://master.d36liwrx5rvjnc.amplifyapp.com/tools/downloads), created by Tim Sevenhuysen.

The original datasets are not redistributed in this repository. Users should obtain the source files directly from the data provider and comply with its usage restrictions.

This project is not affiliated with or endorsed by Oracle's Elixir, Riot Games or the participating esports organisations.

---

**Project focus:** Probabilistic forecasting, chronological model evaluation, data quality, model comparison and interpretation of uncertainty.
