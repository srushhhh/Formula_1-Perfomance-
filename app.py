"""
F1 Performance Intelligence — Streamlit app
Data source: Jolpica-F1 API (api.jolpi.ca), the Ergast-compatible successor to Ergast.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st

# ----------------------------------------------------------------------------
# Page config + theme
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="F1 Performance Intelligence",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_URL = "https://api.jolpi.ca/ergast/f1"

TEAM_COLORS = {
    "Red Bull": "#3671C6", "Ferrari": "#E8002D", "Mercedes": "#27F4D2",
    "McLaren": "#FF8000", "Aston Martin": "#00665F", "Alpine F1 Team": "#FF87BC",
    "Williams": "#64C4FF", "RB F1 Team": "#6692FF", "Kick Sauber": "#00E701",
    "Haas F1 Team": "#B6BABD", "Cadillac F1 Team": "#00693E",
}
DEFAULT_COLOR = "#9CA3AF"

# Official-ish F1 brand palette: race red, near-black, white, carbon grey
F1_RED = "#E10600"
F1_RED_DARK = "#A30500"
F1_BLACK = "#15151E"
F1_CARBON = "#1E1E27"
F1_WHITE = "#FFFFFF"
F1_GREY = "#38383F"
ACCENT_GRADIENT = f"linear-gradient(135deg, {F1_RED} 0%, {F1_RED_DARK} 100%)"

CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@400;600;700;900&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Titillium Web', 'Inter', sans-serif;
    }}
    .stApp {{
        background: {F1_BLACK};
        color: {F1_WHITE};
    }}
    section[data-testid="stSidebar"] {{
        background: {F1_CARBON};
        border-right: 3px solid {F1_RED};
    }}
    /* start/finish checkered stripe under the hero title */
    .f1-stripe {{
        height: 6px;
        width: 100%;
        margin: 6px 0 18px 0;
        background: repeating-linear-gradient(
            90deg,
            {F1_RED} 0px, {F1_RED} 24px,
            {F1_WHITE} 24px, {F1_WHITE} 48px
        );
        border-radius: 3px;
        opacity: 0.9;
    }}
    .glass-card {{
        background: {F1_CARBON};
        border: 1px solid {F1_GREY};
        border-left: 4px solid {F1_RED};
        border-radius: 10px;
        padding: 18px 20px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }}
    .glass-card:hover {{
        transform: translateY(-3px);
        border-color: {F1_RED};
        box-shadow: 0 6px 20px rgba(225,6,0,0.25);
    }}
    .kpi-label {{
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9CA3AF;
        margin-bottom: 4px;
        font-weight: 600;
    }}
    .kpi-value {{
        font-size: 1.65rem;
        font-weight: 900;
        font-style: italic;
        color: {F1_WHITE};
        letter-spacing: 0.01em;
    }}
    .kpi-sub {{
        font-size: 0.82rem;
        color: #9CA3AF;
        margin-top: 2px;
    }}
    h1, h2, h3 {{
        font-family: 'Titillium Web', sans-serif;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.02em;
    }}
    .hero-title {{
        font-size: 2.3rem;
        font-weight: 900;
        font-style: italic;
        text-transform: uppercase;
        letter-spacing: 0.01em;
        color: {F1_WHITE};
        margin-bottom: 0;
    }}
    .hero-title .accent {{
        color: {F1_RED};
    }}
    .hero-sub {{
        color: #9CA3AF;
        margin-top: 2px;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.08em;
    }}
    div[data-testid="stDataFrame"] {{ border-radius: 8px; overflow: hidden; border: 1px solid {F1_GREY}; }}

    /* Tabs styled like an F1 timing-screen menu */
    button[data-baseweb="tab"] {{
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #9CA3AF;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {F1_WHITE} !important;
        border-bottom: 3px solid {F1_RED} !important;
    }}
    div[data-baseweb="tab-highlight"] {{
        background-color: {F1_RED} !important;
    }}

    /* Sidebar widgets */
    .stSelectbox label, .stNumberInput label {{
        color: {F1_WHITE};
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.78rem;
        letter-spacing: 0.05em;
    }}
    section[data-testid="stSidebar"] .stMarkdown h3 {{
        color: {F1_RED};
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def kpi_card(label, value, sub=""):
    st.markdown(
        f"""<div class="glass-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def team_color(team):
    return TEAM_COLORS.get(team, DEFAULT_COLOR)


def hex_to_rgba(hex_color, alpha=0.25):
    """Plotly's `fillcolor` doesn't accept 8-digit hex (hex+alpha) — needs rgba()."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ----------------------------------------------------------------------------
# API layer (cached)
# ----------------------------------------------------------------------------

class F1ApiError(Exception):
    """Raised when the Jolpica API call fails, with the reason preserved for the UI."""
    pass


@st.cache_data(ttl=3600, show_spinner=False)
def api_get(path, params=None):
    url = f"{BASE_URL}/{path}.json"
    try:
        resp = requests.get(url, params=params, timeout=15)
    except requests.exceptions.RequestException as e:
        raise F1ApiError(f"Network error reaching Jolpica API: {e}")
    if resp.status_code == 429:
        raise F1ApiError(
            "Jolpica API rate limit hit (unauthenticated limit is ~200 requests/hour). "
            "Wait a few minutes and try again, or narrow how many rounds get fetched at once."
        )
    if resp.status_code == 404:
        raise F1ApiError(f"No data found at {url} (this season/round/endpoint may not exist).")
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        raise F1ApiError(f"Jolpica API error ({resp.status_code}) for {url}: {e}")
    return resp.json()["MRData"]


@st.cache_data(ttl=3600, show_spinner=False)
def get_season_races(season):
    data = api_get(f"{season}")
    races = data["RaceTable"]["Races"]
    return [(int(r["round"]), r["raceName"], r["date"]) for r in races]


@st.cache_data(ttl=3600, show_spinner=False)
def get_race_results(season, rnd):
    data = api_get(f"{season}/{rnd}/results")
    races = data["RaceTable"]["Races"]
    if not races:
        return None, pd.DataFrame()
    race = races[0]
    rows = []
    for r in race["Results"]:
        grid = r.get("grid", "0")
        rows.append({
            "Position": int(r["position"]),
            "Driver": f'{r["Driver"]["givenName"]} {r["Driver"]["familyName"]}',
            "DriverId": r["Driver"]["driverId"],
            "Code": r["Driver"].get("code", ""),
            "Team": r["Constructor"]["name"],
            "Grid": int(grid) if grid.isdigit() else None,
            "Laps": int(r["laps"]),
            "Status": r["status"],
            "Points": float(r["points"]),
            "FastestLapRank": r.get("FastestLap", {}).get("rank"),
            "FastestLapTime": r.get("FastestLap", {}).get("Time", {}).get("time"),
        })
    df = pd.DataFrame(rows).sort_values("Position").reset_index(drop=True)
    return race, df


@st.cache_data(ttl=3600, show_spinner=False)
def get_qualifying(season, rnd):
    data = api_get(f"{season}/{rnd}/qualifying")
    races = data["RaceTable"]["Races"]
    if not races or "QualifyingResults" not in races[0]:
        return pd.DataFrame()
    rows = []
    for q in races[0]["QualifyingResults"]:
        rows.append({
            "DriverId": q["Driver"]["driverId"],
            "Driver": f'{q["Driver"]["givenName"]} {q["Driver"]["familyName"]}',
            "QualiPosition": int(q["position"]),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def get_standings_through(season, rnd):
    data = api_get(f"{season}/{rnd}/driverStandings")
    lists = data["StandingsTable"]["StandingsLists"]
    if not lists:
        return pd.DataFrame()
    rows = []
    for s in lists[0]["DriverStandings"]:
        if "position" not in s:
            continue
        rows.append({
            "DriverId": s["Driver"]["driverId"],
            "Driver": f'{s["Driver"]["givenName"]} {s["Driver"]["familyName"]}',
            "Team": s["Constructors"][0]["name"],
            "Points": float(s["points"]),
            "Wins": int(s["wins"]),
            "StandingPos": int(s["position"]),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def get_season_progression(season, upto_round):
    frames = []
    for rnd in range(1, upto_round + 1):
        try:
            standings = get_standings_through(season, rnd)
        except F1ApiError as e:
            st.warning(f"Stopped season progression fetch at round {rnd}: {e}")
            break
        if standings.empty:
            continue
        standings["Round"] = rnd
        frames.append(standings)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def get_driver_laps(season, rnd, driver_id):
    """All laps for one driver in one race — used for Consistency Score."""
    try:
        data = api_get(f"{season}/{rnd}/drivers/{driver_id}/laps", params={"limit": 100})
    except Exception:
        return pd.DataFrame()
    races = data["RaceTable"]["Races"]
    if not races or "Laps" not in races[0]:
        return pd.DataFrame()
    rows = []
    for lap in races[0]["Laps"]:
        t = lap["Timings"][0]["time"]
        parts = t.split(":")
        seconds = float(parts[-1]) + (int(parts[-2]) * 60 if len(parts) > 1 else 0)
        rows.append({"Lap": int(lap["number"]), "LapTimeSeconds": seconds})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Custom metrics (not provided by the API — engineered here)
# ----------------------------------------------------------------------------

def consistency_score(laps_df):
    """100 = perfectly even pace, lower = more variable, on an outlier-trimmed lap set."""
    if laps_df.empty or len(laps_df) < 5:
        return None
    q1, q3 = laps_df["LapTimeSeconds"].quantile([0.25, 0.75])
    iqr = q3 - q1
    clean = laps_df[(laps_df["LapTimeSeconds"] >= q1 - 1.5 * iqr) &
                     (laps_df["LapTimeSeconds"] <= q3 + 1.5 * iqr)]
    if len(clean) < 5:
        clean = laps_df
    cv = clean["LapTimeSeconds"].std() / clean["LapTimeSeconds"].mean()
    return round(max(0, 100 - cv * 100 * 8), 1)  # scaled for readable spread


def quali_vs_race_delta(quali_df, results_df):
    merged = results_df.merge(quali_df[["DriverId", "QualiPosition"]], on="DriverId", how="left")
    merged["QualiVsRaceDelta"] = merged["QualiPosition"] - merged["Position"]
    return merged[["Driver", "Team", "QualiPosition", "Position", "QualiVsRaceDelta"]]


def driver_momentum(progression_df, driver_id, upto_round, window=5):
    """EWMA of per-round points scored (not cumulative) over the last `window` rounds."""
    sub = progression_df[progression_df["DriverId"] == driver_id].sort_values("Round")
    sub = sub[sub["Round"] <= upto_round]
    if len(sub) < 2:
        return None
    per_round_points = sub["Points"].diff().fillna(sub["Points"].iloc[0])
    recent = per_round_points.tail(window)
    return round(recent.ewm(span=window).mean().iloc[-1], 1)


def build_radar_metrics(season, rnd, results_df, quali_df):
    """Normalized 0-100 metrics per driver for the radar chart."""
    max_pos = results_df["Position"].max()
    out = []
    for _, row in results_df.iterrows():
        pace_score = 100 * (max_pos - row["Position"] + 1) / max_pos
        quali_row = quali_df[quali_df["DriverId"] == row["DriverId"]]
        quali_pos = quali_row["QualiPosition"].iloc[0] if not quali_row.empty else row["Grid"]
        quali_score = 100 * (max_pos - (quali_pos or max_pos) + 1) / max_pos if quali_pos else 50
        gained = (row["Grid"] or row["Position"]) - row["Position"]
        racecraft_score = float(np.clip(50 + gained * 8, 0, 100))
        reliability_score = 100.0 if row["Status"] in ("Finished",) or "Lap" in str(row["Status"]) else 30.0
        points_score = 100 * row["Points"] / 25 if row["Points"] else 0
        out.append({
            "Driver": row["Driver"], "DriverId": row["DriverId"], "Team": row["Team"],
            "Race Pace": round(pace_score, 1),
            "Qualifying": round(quali_score, 1),
            "Racecraft": round(racecraft_score, 1),
            "Reliability": round(reliability_score, 1),
            "Points Return": round(min(points_score, 100), 1),
        })
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------------

st.sidebar.markdown("### 🏎️ Controls")
season = st.sidebar.number_input("Season", min_value=1950, max_value=2026, value=2025, step=1)

try:
    races = get_season_races(season)
except F1ApiError as e:
    st.error(f"Couldn't load the {season} season: {e}")
    st.stop()
if not races:
    st.error("No race data found for that season.")
    st.stop()

race_labels = [f"Round {r} — {name}" for r, name, _ in races]
race_choice = st.sidebar.selectbox("Race", options=list(range(len(races))),
                                    format_func=lambda i: race_labels[i], index=len(races) - 1)
round_num = races[race_choice][0]

st.sidebar.markdown("---")
st.sidebar.caption("Data: Jolpica-F1 API (Ergast-compatible) · api.jolpi.ca")

# ----------------------------------------------------------------------------
# Load core data for the selected race
# ----------------------------------------------------------------------------

with st.spinner("Loading race data..."):
    try:
        race, results_df = get_race_results(season, round_num)
        quali_df = get_qualifying(season, round_num)
        standings_df = get_standings_through(season, round_num)
        progression_df = get_season_progression(season, round_num)
    except F1ApiError as e:
        st.error(f"Couldn't load data for {season} Round {round_num}: {e}")
        st.stop()

if race is None or results_df.empty:
    st.warning("No results available for this race yet.")
    st.stop()

race_name = race["raceName"]
race_date = race["date"]

st.markdown(f'<div class="hero-title">🏁 {race_name}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="hero-sub">Round <span class="accent" style="color:#E10600">{round_num}</span> '
            f'&nbsp;·&nbsp; {season} Season &nbsp;·&nbsp; {race_date}</div>',
            unsafe_allow_html=True)
st.markdown('<div class="f1-stripe"></div>', unsafe_allow_html=True)
st.write("")

# ----------------------------------------------------------------------------
# KPI row
# ----------------------------------------------------------------------------

winner = results_df.iloc[0]
fastest_lap_row = results_df[results_df["FastestLapRank"] == "1"]
fastest_lap_driver = fastest_lap_row["Driver"].iloc[0] if not fastest_lap_row.empty else "—"
fastest_lap_time = fastest_lap_row["FastestLapTime"].iloc[0] if not fastest_lap_row.empty else ""
results_df["PosGained"] = results_df["Grid"].fillna(results_df["Position"]) - results_df["Position"]
biggest_mover = results_df.loc[results_df["PosGained"].idxmax()]
finishers = (results_df["Status"] == "Finished").sum() + results_df["Status"].str.contains("Lap", na=False).sum()

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Race Winner", winner["Driver"], winner["Team"])
with c2:
    kpi_card("Fastest Lap", fastest_lap_driver, fastest_lap_time)
with c3:
    kpi_card("Biggest Mover", f'{biggest_mover["Driver"]}',
             f'+{int(biggest_mover["PosGained"])} positions')
with c4:
    kpi_card("Classified Finishers", f"{finishers}/{len(results_df)}")

st.write("")

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------

tab_report, tab_champ, tab_radar, tab_compare = st.tabs(
    ["🏁 Race Report", "🏆 Championship", "🎯 Driver Radar", "⚔️ Compare Drivers"]
)

# --- Race Report -------------------------------------------------------
with tab_report:
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Classification")
        podium_style = results_df.style.apply(
            lambda row: ["background-color: rgba(255,215,0,0.15)"] * len(row) if row["Position"] == 1
            else ["background-color: rgba(192,192,192,0.15)"] * len(row) if row["Position"] == 2
            else ["background-color: rgba(205,127,50,0.15)"] * len(row) if row["Position"] == 3
            else [""] * len(row),
            axis=1,
        ).format({"Points": "{:.0f}"})
        st.dataframe(
            podium_style,
            column_order=["Position", "Driver", "Team", "Grid", "Laps", "Status", "Points"],
            hide_index=True, use_container_width=True,
        )

    with right:
        st.subheader("Points scored")
        fig = go.Figure(go.Bar(
            x=results_df["Driver"], y=results_df["Points"],
            marker_color=[team_color(t) for t in results_df["Team"]],
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=380, margin=dict(l=10, r=10, t=20, b=80), xaxis_tickangle=-45,
        )
        st.plotly_chart(fig, use_container_width=True)

    if not quali_df.empty:
        st.subheader("Qualifying vs Race Delta")
        st.caption("Positive = gained places relative to grid; negative = lost places.")
        qvr = quali_vs_race_delta(quali_df, results_df)
        fig2 = go.Figure(go.Bar(
            x=qvr["Driver"], y=qvr["QualiVsRaceDelta"],
            marker_color=["#FFFFFF" if v >= 0 else "#E10600" for v in qvr["QualiVsRaceDelta"]],
        ))
        fig2.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=320, margin=dict(l=10, r=10, t=10, b=80), xaxis_tickangle=-45,
        )
        st.plotly_chart(fig2, use_container_width=True)

# --- Championship -------------------------------------------------------
with tab_champ:
    left, right = st.columns([2, 3])

    with left:
        st.subheader(f"Standings after Round {round_num}")
        st.dataframe(
            standings_df.style.format({"Points": "{:.0f}"}),
            column_order=["StandingPos", "Driver", "Team", "Points", "Wins"],
            hide_index=True, use_container_width=True,
        )

    with right:
        st.subheader("Season points progression (top 10)")
        top10 = standings_df.head(10)["DriverId"].tolist()
        fig3 = go.Figure()
        for did in top10:
            sub = progression_df[progression_df["DriverId"] == did].sort_values("Round")
            if sub.empty:
                continue
            name = sub["Driver"].iloc[-1]
            team = sub["Team"].iloc[-1]
            fig3.add_trace(go.Scatter(
                x=sub["Round"], y=sub["Points"], mode="lines+markers", name=name,
                line=dict(color=team_color(team), width=2), marker=dict(size=4),
            ))
        fig3.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=420, margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.4),
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Driver Momentum")
    st.caption("Exponentially-weighted average of points scored per race over the last 5 rounds — "
               "a form indicator, not just cumulative total.")
    momentum_rows = []
    for _, row in standings_df.head(10).iterrows():
        m = driver_momentum(progression_df, row["DriverId"], round_num)
        momentum_rows.append({"Driver": row["Driver"], "Momentum (pts/race, weighted)": m})
    st.dataframe(pd.DataFrame(momentum_rows), hide_index=True, use_container_width=True)

# --- Driver Radar --------------------------------------------------------
with tab_radar:
    st.subheader("Driver Radar — this race")
    st.caption("Composite 0-100 scores engineered from grid position, finish position, "
               "status, and points — not provided directly by the API.")

    radar_df = build_radar_metrics(season, round_num, results_df, quali_df)
    driver_pick = st.selectbox("Choose a driver", options=radar_df["Driver"].tolist())
    row = radar_df[radar_df["Driver"] == driver_pick].iloc[0]

    categories = ["Race Pace", "Qualifying", "Racecraft", "Reliability", "Points Return"]
    values = [row[c] for c in categories]

    fig4 = go.Figure()
    fig4.add_trace(go.Scatterpolar(
        r=values + [values[0]], theta=categories + [categories[0]],
        fill="toself", line=dict(color=team_color(row["Team"])),
        fillcolor=hex_to_rgba(team_color(row["Team"]), alpha=0.25),
        name=driver_pick,
    ))
    fig4.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=460, margin=dict(l=40, r=40, t=30, b=30),
        showlegend=False,
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.dataframe(radar_df.set_index("Driver")[categories], use_container_width=True)

# --- Compare Drivers ------------------------------------------------------
with tab_compare:
    st.subheader("Head-to-head")
    all_drivers = results_df["Driver"].tolist()
    colA, colB = st.columns(2)
    with colA:
        d1 = st.selectbox("Driver A", all_drivers, index=0, key="d1")
    with colB:
        d2 = st.selectbox("Driver B", all_drivers, index=min(1, len(all_drivers) - 1), key="d2")

    r1 = results_df[results_df["Driver"] == d1].iloc[0]
    r2 = results_df[results_df["Driver"] == d2].iloc[0]

    laps1 = get_driver_laps(season, round_num, r1["DriverId"])
    laps2 = get_driver_laps(season, round_num, r2["DriverId"])
    cs1 = consistency_score(laps1)
    cs2 = consistency_score(laps2)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        kpi_card(f"{d1} — Finish", f'P{r1["Position"]}', r1["Status"])
    with m2:
        kpi_card(f"{d2} — Finish", f'P{r2["Position"]}', r2["Status"])
    with m3:
        kpi_card(f"{d1} — Consistency", cs1 if cs1 is not None else "n/a",
                 "lap-time consistency score")
    with m4:
        kpi_card(f"{d2} — Consistency", cs2 if cs2 is not None else "n/a",
                 "lap-time consistency score")

    if not laps1.empty and not laps2.empty:
        st.subheader("Lap time comparison")
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=laps1["Lap"], y=laps1["LapTimeSeconds"],
                                   mode="lines", name=d1, line=dict(color=team_color(r1["Team"]))))
        fig5.add_trace(go.Scatter(x=laps2["Lap"], y=laps2["LapTimeSeconds"],
                                   mode="lines", name=d2, line=dict(color=team_color(r2["Team"]))))
        fig5.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Lap time (s)",
        )
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("Lap-by-lap data unavailable for one or both drivers for this race.")

st.write("")
st.caption("Built on the Jolpica-F1 API (Ergast-compatible) · api.jolpi.ca · "
           "All composite metrics (Consistency Score, Momentum, Racecraft, etc.) are engineered "
           "in this app, not provided by the source API.")