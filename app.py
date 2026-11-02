import streamlit as st
import pandas as pd
from elo_model import elo_probability
from series_probability import series_score_probabilities, series_win_probability

st.set_page_config(page_title="LoL Win Probability Predictor", page_icon="🏆", layout="centered")

@st.cache_data
def load_domestic_ratings():
    return pd.read_csv("latest_team_ratings.csv")

@st.cache_data
def load_international_ratings():
    return pd.read_csv("international_team_ratings.csv")

@st.cache_data
def load_backtest_summary():
    return pd.read_csv("lol_elo_model_comparison.csv")

@st.cache_data
def load_match_dates():
    """Optional: only use dates from an actual exported match-level file."""
    try:
        dates = pd.read_csv("lol_match_predictions_final.csv", usecols=["date", "league"])
        dates["date"] = pd.to_datetime(dates["date"], errors="coerce")
        return dates
    except (FileNotFoundError, ValueError):
        return None


def show_data_as_of(league=None):
    dates = load_match_dates()
    if dates is None:
        st.caption("Rating snapshot date: not independently verified. The published rating CSV does not include an as-of date.")
        return
    if league is not None:
        dates = dates[dates["league"] == league]
    last_date = dates["date"].max()
    if pd.isna(last_date):
        st.caption("Rating snapshot date: unavailable in the supplied match data.")
    else:
        st.caption(f"Latest game date in exported source data: {last_date:%Y-%m-%d}. Ratings are a historical snapshot, not live estimates.")


def show_prediction(team_a, team_b, elo_a, elo_b, prob_a, side_labels=False):
    prob_b = 1 - prob_a
    st.subheader("Estimated Single-Game Win Probability")
    left, right = st.columns(2)
    with left:
        st.metric(f"{'🔵 Blue · ' if side_labels else ''}{team_a}", f"{prob_a:.1%}")
        st.caption(f"Elo rating: {elo_a:.0f}")
    with right:
        st.metric(f"{'🔴 Red · ' if side_labels else ''}{team_b}", f"{prob_b:.1%}")
        st.caption(f"Elo rating: {elo_b:.0f}")
    st.progress(float(prob_a))
    st.caption(f"{team_a}: {prob_a:.1%}  |  {team_b}: {prob_b:.1%}")
    st.markdown("**Win probability comparison**")
    plot_df = pd.DataFrame({"Probability": [prob_a * 100, prob_b * 100]}, index=[team_a, team_b])
    st.bar_chart(plot_df, horizontal=True, height=210, x_label="Win probability (%)")
    st.caption(f"Pre-match Elo difference ({team_a} − {team_b}): {elo_a - elo_b:+.1f}")


def show_series_prediction(team_a, team_b, prob_a):
    st.divider()
    st.subheader("Series Win Probability")
    best_of = st.radio("Series format", [1, 3, 5], format_func=lambda x: f"Bo{x}", horizontal=True)
    if best_of == 1:
        st.metric(f"{team_a} wins Bo1", f"{prob_a:.1%}")
    else:
        series_prob = series_win_probability(prob_a, best_of)
        scores = series_score_probabilities(prob_a, best_of)
        c1, c2 = st.columns(2)
        c1.metric(f"{team_a} wins Bo{best_of}", f"{series_prob:.1%}")
        c2.metric(f"{team_b} wins Bo{best_of}", f"{1 - series_prob:.1%}")
        needed = best_of // 2 + 1
        ordered_scores = sorted(scores.items(), key=lambda item: (-int(item[0].split('-')[0]), int(item[0].split('-')[1])))
        score_df = pd.DataFrame([
            {"Result": f"{team_a} {score.replace('-', '–')} {team_b}", "Probability": probability}
            for score, probability in ordered_scores
        ])
        st.markdown("**Exact series score probabilities**")
        st.dataframe(score_df.style.format({"Probability": "{:.1%}"}), hide_index=True, use_container_width=True)
    st.caption("Model assumptions: games are independent and each game has a constant win probability. "
               "The calculation does not account for changing side selection, drafting, roster changes, "
               "patches, adaptation, or game-to-game dependence. Series outcomes have not been separately backtested.")


def show_historical_performance():
    st.header("Historical Model Evaluation · Domestic V1–V3")
    st.write("Chronological, pre-game Elo predictions evaluated against recorded LCK and LEC game results.")
    summary = load_backtest_summary().copy()
    required = {"Year", "Model", "Matches", "Accuracy", "Log Loss", "Brier Score"}
    if not required.issubset(summary.columns):
        st.error("The backtest CSV is missing required columns.")
        return
    summary["Year"] = pd.to_numeric(summary["Year"], errors="coerce")
    for col in ["Matches", "Accuracy", "Log Loss", "Brier Score"]:
        summary[col] = pd.to_numeric(summary[col], errors="coerce")
    summary = summary.dropna(subset=["Year", "Model", "Accuracy", "Log Loss", "Brier Score"])
    years = sorted(summary["Year"].astype(int).unique().tolist(), reverse=True)
    if not years:
        st.error("No valid evaluation data found.")
        return
    year = st.selectbox("Evaluation year", years)
    year_df = summary[summary["Year"] == year].copy().sort_values("Model")
    v3 = year_df[year_df["Model"] == "V3"]
    if not v3.empty:
        v3row = v3.iloc[0]
        a, b, c = st.columns(3)
        a.metric("V3 accuracy", f"{v3row['Accuracy']:.2%}")
        b.metric("V3 Brier score ↓", f"{v3row['Brier Score']:.4f}")
        c.metric("V3 log loss ↓", f"{v3row['Log Loss']:.4f}")
        st.caption(f"Evaluated on {int(v3row['Matches']):,} individual games in {year}. Accuracy uses a 50% classification threshold.")
    display = year_df[["Model", "Matches", "Accuracy", "Log Loss", "Brier Score"]].copy()
    st.dataframe(display.style.format({"Matches": "{:,.0f}", "Accuracy": "{:.2%}", "Log Loss": "{:.4f}", "Brier Score": "{:.4f}"}), hide_index=True, use_container_width=True)
    st.markdown("**Accuracy comparison**")
    st.bar_chart(year_df.set_index("Model")["Accuracy"] * 100, y_label="Accuracy (%)")
    st.markdown("**Probability-quality comparison**")
    st.bar_chart(year_df.set_index("Model")[["Brier Score", "Log Loss"]])
    st.info("Interpretation: Accuracy measures how often the predicted winner was correct. "
            "Brier score and log loss evaluate probability quality; lower is better. "
            "These are historical results, not guarantees for upcoming matches.")
    st.warning("Evaluation limitation: 2025 outcomes were used in model development, and 2026 outcomes "
               "were inspected during development. These results are exploratory chronological backtests, "
               "not a fully untouched holdout or live predictive-performance estimate. "
               "They evaluate domestic single-game predictions (V1–V3), not international V4 or Bo3/Bo5 series forecasts.")


st.title("🏆 LoL Esports Win Probability Predictor")
st.write("Historical Elo-based game and series estimates — independent portfolio project.")
page = st.radio("View", ["🏠 Domestic Leagues (V3)", "🌍 International Matchups (V4)", "📊 Historical Model Evaluation"], horizontal=False)
st.divider()

if page == "🏠 Domestic Leagues (V3)":
    df = load_domestic_ratings()
    st.header("Domestic Leagues · V3")
    st.write("Compare first-team participants within the same LCK or LEC league.")
    league = st.selectbox("Select League", sorted(df["league"].dropna().unique()))
    league_df = df[df["league"] == league].copy()
    # The published LEC ratings contain an academy team. Keep source data unchanged.
    academy_exclusions = {"Karmine Corp Blue"}
    excluded = sorted(set(league_df["team"].dropna()).intersection(academy_exclusions))
    league_df = league_df[~league_df["team"].isin(academy_exclusions)]
    if excluded:
        st.caption("Academy entries excluded from team selection: " + ", ".join(excluded) + ". Source CSV remains unchanged.")
    teams = sorted(league_df["team"].dropna().unique())
    if len(teams) < 2:
        st.error("At least two eligible teams are required.")
        st.stop()
    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Team A", teams, index=0, key="v3_a")
    with col2:
        team_b = st.selectbox("Team B", [t for t in teams if t != team_a], index=0, key="v3_b")
    elo_a = float(league_df.loc[league_df["team"] == team_a, "elo_rating"].iloc[0])
    elo_b = float(league_df.loc[league_df["team"] == team_b, "elo_rating"].iloc[0])
    prob_a = elo_probability(elo_a, elo_b)
    show_data_as_of(league)
    st.divider()
    show_prediction(team_a, team_b, elo_a, elo_b, prob_a)
    show_series_prediction(team_a, team_b, prob_a)
    st.caption("V3: Elo K=24, seasonal carryover=0.75, no side adjustment; league-specific ratings.")

elif page == "🌍 International Matchups (V4)":
    df = load_international_ratings()
    st.header("International Matchups · V4")
    st.write("Compare teams from different regions on a shared global Elo scale.")
    st.info("Blue/Red side is explicitly assigned. Blue receives a +20 Elo prediction adjustment.")
    regions = sorted(df["region"].dropna().unique())
    col1, col2 = st.columns(2)
    with col1:
        blue_region = st.selectbox("🔵 Blue region", regions, key="v4_blue_region")
    with col2:
        red_region = st.selectbox("🔴 Red region", [r for r in regions if r != blue_region], key="v4_red_region")
    blue_teams = sorted(df.loc[df["region"] == blue_region, "team"].dropna().unique())
    red_teams = sorted(df.loc[df["region"] == red_region, "team"].dropna().unique())
    col1, col2 = st.columns(2)
    with col1:
        blue_team = st.selectbox("🔵 Blue team", blue_teams, key="v4_blue_team")
    with col2:
        red_team = st.selectbox("🔴 Red team", red_teams, key="v4_red_team")
    elo_blue = float(df.loc[(df["region"] == blue_region) & (df["team"] == blue_team), "elo"].iloc[0])
    elo_red = float(df.loc[(df["region"] == red_region) & (df["team"] == red_team), "elo"].iloc[0])
    prob_blue = elo_probability(elo_blue, elo_red, blue_bonus=20)
    st.caption("International rating snapshot date: unverified; CSV contains no timestamp.")
    st.divider()
    show_prediction(blue_team, red_team, elo_blue, elo_red, prob_blue, side_labels=True)
    show_series_prediction(blue_team, red_team, prob_blue)
    no_bonus = elo_probability(elo_blue, elo_red)
    st.caption(f"Without Blue adjustment: {blue_team} {no_bonus:.1%} · With +20 Elo: {prob_blue:.1%}")
    st.caption("V4: global team Elo, K=24, seasonal carryover=0.75; Blue +20 affects predictions only.")
    st.warning("Experimental international model; no reported independent V4 backtest. "
               "Regions are approximate and some ratings may be from earlier seasons. "
               "Cross-region Elo comparisons may be unreliable without international connections.")

else:
    show_historical_performance()

st.divider()
st.caption("Data source: Oracle's Elixir (Tim Sevenhuysen). Independent, non-commercial portfolio project. "
           "Not endorsed by Riot Games or Oracle's Elixir. Not intended for betting or gambling.")
