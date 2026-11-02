import streamlit as st
import pandas as pd
from elo_model import elo_probability
from series_probability import (
    series_score_probabilities,
    series_win_probability,
)

st.set_page_config(
    page_title="LoL Win Probability Predictor",
    page_icon="🏆",
    layout="centered",
)

@st.cache_data
def load_domestic_ratings():
    return pd.read_csv("latest_team_ratings.csv")

@st.cache_data
def load_international_ratings():
    return pd.read_csv("international_team_ratings.csv")




def show_prediction(team_a, team_b, elo_a, elo_b, prob_a, side_labels=False):
    prob_b = 1 - prob_a
    st.subheader("Estimated Match Win Probability")
    left, right = st.columns(2)
    with left:
        st.metric(f"{'🔵 Blue · ' if side_labels else ''}{team_a}", f"{prob_a:.1%}")
        st.caption(f"Elo rating: {elo_a:.0f}")
    with right:
        st.metric(f"{'🔴 Red · ' if side_labels else ''}{team_b}", f"{prob_b:.1%}")
        st.caption(f"Elo rating: {elo_b:.0f}")

    # Original-style probability progress indicator
    st.progress(float(prob_a))
    st.caption(f"{team_a}: {prob_a:.1%}  |  {team_b}: {prob_b:.1%}")

    # New comparison visual, using only Streamlit and pandas
    st.markdown("**Win probability comparison**")
    plot_df = pd.DataFrame(
        {"Probability": [prob_a * 100, prob_b * 100]},
        index=[team_a, team_b],
    )
    st.bar_chart(plot_df, horizontal=True, height=210, x_label="Win probability (%)")

    elo_delta = elo_a - elo_b
    st.caption(f"Pre-match Elo difference ({team_a} − {team_b}): {elo_delta:+.1f}")


st.title("🏆 LoL Esports Win Probability Predictor")
st.write("Compare historical Elo-based match win probability estimates.")

page = st.radio(
    "Prediction mode",
    ["🏠 Domestic Leagues (V3)", "🌍 International Matchups (V4)"],
    horizontal=True,
)
st.divider()

if page == "🏠 Domestic Leagues (V3)":
    df = load_domestic_ratings()
    st.header("Domestic Leagues · V3")
    st.write("Compare teams within the same LCK or LEC league.")

    league = st.selectbox("Select League", sorted(df["league"].dropna().unique()))
    league_df = df[df["league"] == league]
    teams = sorted(league_df["team"].dropna().unique())

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Team A", teams, index=0, key="v3_a")
    with col2:
        available_teams = [team for team in teams if team != team_a]
        team_b = st.selectbox("Team B", available_teams, index=0, key="v3_b")

    elo_a = float(league_df.loc[league_df["team"] == team_a, "elo_rating"].iloc[0])
    elo_b = float(league_df.loc[league_df["team"] == team_b, "elo_rating"].iloc[0])
    prob_a = elo_probability(elo_a, elo_b)

    st.divider()
    show_prediction(team_a, team_b, elo_a, elo_b, prob_a)
    st.caption("V3 model: Elo K=24, seasonal carryover=0.75, no side adjustment. League-specific ratings.")

else:
    df = load_international_ratings()
    st.header("International Matchups · V4")
    st.write("Compare teams from different regions on a shared global Elo scale.")
    st.info("Blue/Red side is explicitly assigned. The Blue side receives a +20 Elo prediction adjustment.")

    # Select a region first; restrict the opposing region to a different group.
    regions = sorted(df["region"].dropna().unique())
    col1, col2 = st.columns(2)
    with col1:
        blue_region = st.selectbox("🔵 Blue region", regions, key="v4_blue_region")
    with col2:
        red_regions = [region for region in regions if region != blue_region]
        red_region = st.selectbox("🔴 Red region", red_regions, key="v4_red_region")

    blue_teams = sorted(df.loc[df["region"] == blue_region, "team"].dropna().unique())
    red_teams = sorted(df.loc[df["region"] == red_region, "team"].dropna().unique())

    team_col1, team_col2 = st.columns(2)
    with team_col1:
        blue_team = st.selectbox("🔵 Blue team", blue_teams, key="v4_blue_team")
    with team_col2:
        red_team = st.selectbox("🔴 Red team", red_teams, key="v4_red_team")

    elo_blue = float(df.loc[df["team"] == blue_team, "elo"].iloc[0])
    elo_red = float(df.loc[df["team"] == red_team, "elo"].iloc[0])
    prob_blue = elo_probability(elo_blue, elo_red, blue_bonus=20)

    st.divider()
    show_prediction(blue_team, red_team, elo_blue, elo_red, prob_blue, side_labels=True)

    # Side adjustment is a prediction-only correction, not a rating update.
    no_bonus = elo_probability(elo_blue, elo_red)
    st.caption(f"Without Blue-side adjustment: {blue_team} {no_bonus:.1%} · With +20 Elo: {prob_blue:.1%}")
    st.caption("V4 model: global team Elo, K=24, seasonal carryover=0.75. "
               "Blue +20 Elo applied to prediction only (not Elo updates).")
    st.warning("Experimental historical model, not live odds. Regions are approximate, "
               "some team ratings are from earlier seasons, and cross-region Elo comparisons "
               "may be unreliable for teams without international connections.")

st.divider()
st.caption("Data source: Oracle's Elixir (Tim Sevenhuysen). Independent, non-commercial portfolio project. "
           "Not endorsed by Riot Games or Oracle's Elixir. Not intended for betting or gambling.")
