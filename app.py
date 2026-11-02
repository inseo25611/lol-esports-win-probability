import streamlit as st
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

from elo_model import elo_probability
from series_probability import series_score_probabilities, series_win_probability

st.set_page_config(page_title="LoL Win Probability Predictor", page_icon="🏆", layout="centered")

@st.cache_data
def load_domestic_ratings():
    return validate_ratings(pd.read_csv(DATA_DIR / "latest_team_ratings.csv"))

@st.cache_data
def load_international_ratings():
    return validate_ratings(pd.read_csv(DATA_DIR / "international_team_ratings.csv"), international=True)

@st.cache_data
def load_backtest_summary():
    return pd.read_csv(DATA_DIR / "lol_elo_model_comparison.csv")

@st.cache_data
def read_optional_csv(path):
    try:
        df = pd.read_csv(DATA_DIR / path)
        if path == "lol_match_predictions_final.csv":
            validate_match_predictions(df)
        elif path == "v4_match_predictions_final.csv":
            validate_match_predictions(df, international=True)
        return df
    except FileNotFoundError:
        return None


def compute_metrics(data, p_col):
    import numpy as np
    y = data["blue_win"].to_numpy(dtype=float)
    p = data[p_col].to_numpy(dtype=float)
    pc = np.clip(p, 1e-15, 1 - 1e-15)
    return {"Matches": len(y), "Accuracy": ((p >= 0.5) == y).mean(),
            "Brier Score": ((p - y)**2).mean(),
            "Log Loss": (-y*np.log(pc) - (1-y)*np.log(1-pc)).mean()}


def show_calibration(data, p_col, title):
    import numpy as np
    if len(data) < 10:
        st.info("Not enough games for a meaningful calibration curve.")
        return
    vals = data[[p_col,"blue_win"]].copy().dropna()
    vals["bin"] = pd.qcut(vals[p_col], min(5, len(vals)), labels=False, duplicates="drop")
    c = vals.groupby("bin",observed=True).agg(
        predicted=(p_col,"mean"), observed=("blue_win","mean"), games=("blue_win","size")
    ).reset_index(drop=True)
    st.markdown(f"**{title}**")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6,4.5))
    ax.plot([0,1],[0,1],linestyle="--",label="Perfect calibration")
    ax.plot(c["predicted"],c["observed"],marker="o",label="Observed win rate")
    ax.set(xlim=(0,1),ylim=(0,1),xlabel="Mean predicted Blue win probability",ylabel="Observed Blue win rate")
    ax.legend();ax.grid(alpha=.2)
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(c.rename(columns={"predicted":"Mean predicted","observed":"Observed win rate","games":"Games"}).style.format({"Mean predicted":"{:.1%}","Observed win rate":"{:.1%}"}), hide_index=True)
    st.caption("Five approximately equally sized probability groups. The counts indicate the reliability of each point; this is descriptive, not an independent validation.")


ACADEMY_TEAMS = {"Karmine Corp Blue"}

# Snapshot-based first-team selection only; this is NOT a live 2026 roster source.
FIRST_TEAM_SELECTION = {'Korea': ['Gen.G', 'Hanwha Life Esports', 'Dplus Kia', 'T1', 'KT Rolster', 'BNK FEARX', 'HANJIN BRION', 'Nongshim RedForce', 'DN SOOPers', 'Kiwoom DRX'], 'China': ['Bilibili Gaming', 'Invictus Gaming', 'Top Esports', 'Ninjas in Pyjamas', "Anyone's Legend", 'JD Gaming', 'LGD Gaming', 'Team WE', 'Weibo Gaming', 'LNG Esports', 'EDward Gaming', 'ThunderTalk Gaming', 'Ultra Prime', 'Oh My God'], 'Europe': ['Karmine Corp', 'G2 Esports', 'Movistar KOI', 'Team Vitality', 'GiantX', 'Natus Vincere', 'Fnatic', 'SK Gaming', 'Shifters', 'Team Heretics'], 'North America': ['LYON', 'Team Liquid', 'Cloud9', 'FlyQuest', 'Shopify Rebellion', 'Dignitas'], 'South America': ['LØS', 'FURIA', 'LOUD', 'RED Canids', 'Vivo Keyd Stars', 'paiN Gaming', 'Fluxo W7M'], 'Asia Pacific': ['CTBC Flying Oyster', 'Frank Esports', 'GAM Esports', 'MVK Esports', 'Team Secret Whales', 'Fukuoka SoftBank HAWKS gaming', 'DetonatioN FocusMe', 'Ground Zero Gaming']}


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



def show_new_evaluation():
    st.header("Historical Probability Evaluation")
    tab_d, tab_v, tab_s = st.tabs(["Domestic V1–V3", "International V4", "Bo3 / Bo5"])
    with tab_d:
        results = read_optional_csv("domestic_model_performance.csv")
        games = read_optional_csv("lol_match_predictions_final.csv")
        if results is None:
            st.error("Domestic evaluation data unavailable.")
        else:
            year = st.selectbox("Season", sorted(results.Year.unique(),reverse=True),key="eval_d_year")
            league = st.selectbox("League",sorted(results.League.unique()),key="eval_d_league")
            r=results[(results.Year==year)&(results.League==league)].copy().sort_values("Model")
            st.dataframe(r[["Model","Matches","Accuracy","Brier Score","Log Loss"]].style.format({"Accuracy":"{:.2%}","Brier Score":"{:.4f}","Log Loss":"{:.4f}"}),hide_index=True)
            st.bar_chart(r.set_index("Model")["Accuracy"]*100,y_label="Accuracy (%)")
            if games is not None:
                model=st.selectbox("Calibration model (domestic only)",["V3","V2","V1"],key="eval_d_model")
                g=games[games.year==year]
                if league!="All":g=g[g.league==league]
                show_calibration(g, f"{model.lower()}_blue_prob",f"{model} calibration · {year} · {league}")
                st.caption("For International V4 calibration, open the International V4 tab and provide its game-level export.")
            else:
                st.info("Game-level calibration data unavailable.")
            st.caption("Historical game-level predictions, evaluated pre-match. Development data were examined when models were selected; treat results as exploratory.")
    with tab_v:
        v=read_optional_csv("v4_match_predictions_final.csv")
        if v is None:
            st.info("V4 game-level evaluation data unavailable.")
        else:
            v=v[v.cross_region.astype(str).str.lower().isin(["true","1","1.0"])].copy()
            v["date"]=pd.to_datetime(v.date,errors="coerce")
            if v.empty:
                st.warning("No verified cross-regional V4 matches in exported data.")
            else:
                yr=st.selectbox("Year",sorted(v.year.dropna().unique(),reverse=True),key="eval_v_year")
                g=v[v.year==yr].dropna(subset=["blue_win","v4_blue_prob","v4_adjusted_prob"])
                rows=[]
                for label,col in [("V4 baseline","v4_blue_prob"),("V4 +20 Blue","v4_adjusted_prob")]:
                    rows.append({"Model":label,**compute_metrics(g,col)})
                df=pd.DataFrame(rows)
                st.dataframe(df.style.format({"Accuracy":"{:.2%}","Brier Score":"{:.4f}","Log Loss":"{:.4f}"}),hide_index=True)
                choice=st.radio("Calibration",["V4 baseline","V4 +20 Blue"],horizontal=True)
                show_calibration(g,"v4_blue_prob" if choice=="V4 baseline" else "v4_adjusted_prob",f"{choice} calibration · {yr}")
                st.caption("Cross-region international games only. Pre-match Elo, +20 used for forecasts only; within-series successive games may be correlated.")
    with tab_s:
        st.caption("Exploratory V4 series · pre-series neutral Elo probability")
        series = read_optional_csv("series_exploratory_backtest.csv")
        if series is None:
            st.info("Series evaluation data unavailable.")
        else:
            series = series.dropna(subset=["series_prob_a", "result_a"]).copy()
            series["best_of"] = pd.to_numeric(series["best_of"], errors="coerce")
            choices = sorted(series["best_of"].dropna().astype(int).unique())
            if not choices:
                st.warning("No evaluated series available.")
            else:
                fmt = st.selectbox("Series format", ["All", *choices], format_func=lambda x: "All (Bo3 + Bo5)" if x == "All" else f"Bo{x}",key="final_format")
                subset = series.copy() if fmt == "All" else series[series.best_of == fmt].copy()
                year = st.selectbox("Period", ["All"] + sorted(subset.year.unique().tolist(), reverse=True),key="final_period")
                if year != "All": subset = subset[subset.year == year]
                st.caption("Selected small series sample; public source links are recorded but row-level verification remains undocumented.")
                if subset.empty:
                    st.info("No crosschecked series in this period.")
                else:
                    eval_df = subset.rename(columns={"result_a":"blue_win", "series_prob_a":"series_prob"})
                    m = compute_metrics(eval_df,"series_prob")
                    st.caption(f"Log loss: model {m['Log Loss']:.4f} vs constant 50/50 baseline 0.6931 (lower is better).")
                    a,b,c,d=st.columns(4)
                    a.metric("Series in sample",f"{m['Matches']}")
                    b.metric("Accuracy",f"{m['Accuracy']:.1%}")
                    c.metric("Brier ↓",f"{m['Brier Score']:.4f}")
                    d.metric("Log loss ↓",f"{m['Log Loss']:.4f}")
                    st.dataframe(subset[["date","competition","team_a","team_b","score","neutral_game_prob_a","series_prob_a","result_a"]].rename(columns={"neutral_game_prob_a":"Neutral single-game P(A)","series_prob_a":"Pre-series P(A)","result_a":"Team A won"}).style.format({"Neutral single-game P(A)":"{:.1%}","Pre-series P(A)":"{:.1%}"}),hide_index=True)
                    if len(subset)>=15:
                        show_calibration(eval_df,"series_prob",f"Exploratory {fmt} calibration · {year}")
                    else:
                        st.caption("Calibration is not shown for fewer than 15 series.")
                st.caption("Method: pre-series Elo, neutral side, constant independent game probability. Selected series only.")

@st.cache_data
def load_match_dates():
    """Optional: only use dates from an actual exported match-level file."""
    try:
        dates = pd.read_csv(DATA_DIR / "lol_match_predictions_final.csv", usecols=["date", "league"])
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
               "patches, adaptation, or game-to-game dependence. Limited exploratory series backtesting is available in Historical Model Evaluation; it is not an independently held-out evaluation.")


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
    academy_exclusions = ACADEMY_TEAMS  # verified exclusion; keep historical training data intact
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
    st.caption("Blue +20 Elo adjustment applies to single-game forecasts.")
    # Exclude known academy teams from interactive selections only, not from historical data.
    original_count = len(df)
    df = filter_first_team_ratings(df)
    st.caption(f"Showing {len(df)} conservatively selected first-team candidates "
               f"from the 2026 rating snapshot (out of {original_count} historical entries). "
               "This is not a live roster list; source CSV and historical evaluations are unchanged.")
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
    st.caption("International rating snapshot date: unverified; CSV contains no timestamp. See V4 backtesting for its latest match date.")
    st.divider()
    show_prediction(blue_team, red_team, elo_blue, elo_red, prob_blue, side_labels=True)
    show_series_prediction(blue_team, red_team, prob_blue)
    no_bonus = elo_probability(elo_blue, elo_red)
    st.caption(f"Without Blue adjustment: {blue_team} {no_bonus:.1%} · With +20 Elo: {prob_blue:.1%}")
    st.caption("V4: global team Elo, K=24, seasonal carryover=0.75; Blue +20 affects predictions only.")


else:
    show_new_evaluation()

with st.expander("Model limitations"):
    st.write("Exploratory historical estimates, not untouched holdout tests. Series results use a small selected sample with documented source links but no independent row-level verification. The series formula assumes constant independent game probabilities; future game-side assignments are not used in pre-series evaluation.")
st.divider()
st.caption("Data source: Oracle's Elixir (Tim Sevenhuysen). Independent, non-commercial portfolio project. "
           "Not endorsed by Riot Games or Oracle's Elixir. Not intended for betting or gambling.")
