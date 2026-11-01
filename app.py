
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="LoL Win Probability Predictor",
    page_icon="🏆",
    layout="centered"
)

# Load team Elo ratings
@st.cache_data
def load_ratings():
    return pd.read_csv("latest_team_ratings.csv")

df = load_ratings()

st.title("🏆 LoL Esports Win Probability Predictor")

st.write(
    "Explore estimated match win probabilities "
    "using Elo ratings from LCK and LEC matches."
)

# Select league
league = st.selectbox(
    "Select League",
    sorted(df["league"].unique())
)

# Get teams in the selected league
league_df = df[df["league"] == league]

teams = sorted(league_df["team"].unique())

# Select two different teams
col1, col2 = st.columns(2)

with col1:
    team_a = st.selectbox(
        "Team A",
        teams,
        index=0
    )

with col2:
    available_teams = [
        team for team in teams if team != team_a
    ]

    team_b = st.selectbox(
        "Team B",
        available_teams,
        index=0
    )

# Retrieve Elo ratings
elo_a = float(
    league_df.loc[
        league_df["team"] == team_a,
        "elo_rating"
    ].iloc[0]
)

elo_b = float(
    league_df.loc[
        league_df["team"] == team_b,
        "elo_rating"
    ].iloc[0]
)

# Calculate win probabilities
prob_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
prob_b = 1 - prob_a

st.divider()

st.subheader("Estimated Match Win Probability")

left, right = st.columns(2)

with left:
    st.metric(team_a, f"{prob_a:.1%}")
    st.caption(f"Elo rating: {elo_a:.0f}")

with right:
    st.metric(team_b, f"{prob_b:.1%}")
    st.caption(f"Elo rating: {elo_b:.0f}")

st.progress(prob_a)

st.caption(
    f"Probability of {team_a} winning: {prob_a:.1%} | "
    f"{team_b} winning: {prob_b:.1%}"
)

st.divider()

st.caption(
    "Model: Elo K=24, seasonal carryover=0.75. "
    "Ratings reflect historical matches through September 2026."
)

st.caption(
    "Data source: Oracle's Elixir (Tim Sevenhuysen). "
    "Independent, non-commercial portfolio project. "
    "Not endorsed by Riot Games or Oracle's Elixir. "
    "Not intended for betting or gambling."
)
