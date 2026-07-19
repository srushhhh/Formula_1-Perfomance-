# F1 Performance Intelligence — Streamlit App

Live F1 analytics dashboard built on the [Jolpica-F1 API](https://github.com/jolpica/jolpica-f1) (the Ergast-compatible successor to Ergast).

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then pick a season and race in the sidebar.

## What's inside

- **Glassmorphic F-1 theme** — custom CSS, animated gradient KPI cards, no default Streamlit look.
- **Race Report tab** — podium-highlighted classification, points-scored chart, Qualifying vs Race Delta chart.
- **Championship tab** — live standings, top-10 season points progression line chart, and a **Driver Momentum** metric (EWMA of points scored over the last 5 rounds — a form indicator, not just cumulative points).
- **Driver Radar tab** — a 5-axis radar (Race Pace, Qualifying, Racecraft, Reliability, Points Return) computed per driver from grid/finish/points data. All scores are engineered in `app.py`, not returned by the API.
- **Compare Drivers tab** — head-to-head finish comparison plus a **Consistency Score** (outlier-trimmed lap-time coefficient of variation) and an overlaid lap-time chart, pulled from per-driver lap data.

## Notes

- Data is cached for 1 hour per query (`st.cache_data`) to stay well under the API's unauthenticated rate limit (~200 req/hr).
- Lap-time data (Consistency Score, lap chart) isn't available for every historical race — the app degrades gracefully to "n/a" / an info message when it's missing.
- To deploy: push to GitHub and connect the repo on [Streamlit Community Cloud](https://streamlit.io/cloud) — no extra config needed beyond `requirements.txt`.

## Next steps (see the full playbook)

This implements the top few items from the ranked feature list — design system, custom metrics, radar chart, championship progression, driver comparison. Natural next additions: ML winner/podium prediction with SHAP, the Fantasy F1 optimizer, and the LLM-grounded query layer.
