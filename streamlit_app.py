"""National Databall Association — NBA Prediction System

Streamlit demo app displaying model predictions, comparisons,
calibration analysis, backtesting results, and bet tracking.

Usage:
    streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timezone, timedelta

# ── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="National Databall Association",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar Navigation ──────────────────────────────────────────────────
st.sidebar.title("🏀 National Databall Association")
st.sidebar.caption("NBA Prediction System — Calibration-Optimized ML")

page = st.sidebar.radio(
    "Navigate",
    ["Today's Predictions", "Model Comparison", "Backtesting", "Bet Tracker", "About"],
)

# ── Helper Functions ─────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_strategy_params():
    """Load saved strategy parameters."""
    path = Path("models/strategy_params.joblib")
    if path.exists():
        return joblib.load(path)
    return {}


@st.cache_data(ttl=60)
def load_bet_tracker():
    """Load bet tracker CSV."""
    path = Path("data/bet_tracker.csv")
    if path.exists():
        df = pd.read_csv(path)
        if len(df) > 0:
            return df
    return pd.DataFrame()


def american_to_decimal(odds: int) -> float:
    if odds > 0:
        return odds / 100.0 + 1.0
    elif odds < 0:
        return 100.0 / abs(odds) + 1.0
    return 2.0


def get_predictions():
    """Run the prediction pipeline and return results."""
    from app.data.storage import DataStore
    from app.ml.features.enhanced_features import EnhancedFeatureBuilder
    from app.data.collectors.odds_api_collector import OddsAPICollector, LivePredictor

    store = DataStore()
    all_seasons = []
    for season in [2019, 2020, 2021, 2022, 2023, 2024, 2025]:
        df = store.load_season_games(season)
        if not df.empty:
            all_seasons.append(df)
    all_games = pd.concat(all_seasons, ignore_index=True)

    builder = EnhancedFeatureBuilder(all_games, scrape_four_factors=False)
    predictor = LivePredictor()
    collector = OddsAPICollector(api_key="f08a75c6c952195ea0fb6badab1e631b")
    odds_games = collector.get_current_odds()

    if not odds_games:
        return [], collector.remaining_requests

    predictions = []
    for game in odds_games:
        home_matches = all_games[all_games["home_team_abbr"] == game.home_abbr]
        away_matches = all_games[all_games["away_team_abbr"] == game.away_abbr]

        if len(home_matches) == 0 or len(away_matches) == 0:
            continue

        home_id = home_matches.iloc[-1]["home_team_id"]
        away_id = away_matches.iloc[-1]["away_team_id"]
        latest_date = all_games["game_date"].max()

        fake_game = pd.Series({
            "game_id": game.game_id,
            "game_date": latest_date,
            "season": 2025,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_team_abbr": game.home_abbr,
            "away_team_abbr": game.away_abbr,
            "home_score": 0,
            "away_score": 0,
            "home_win": False,
        })

        features = builder._build_game_features(fake_game)
        if features is None:
            continue

        prediction = predictor.predict_game(game, features)
        predictions.append(prediction)

    return predictions, collector.remaining_requests


# ══════════════════════════════════════════════════════════════════════════
# PAGE 1: TODAY'S PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════

if page == "Today's Predictions":
    st.title("Today's NBA Predictions")
    st.caption("Live odds from The Odds API • Predictions from calibration-optimized Random Forest")

    if st.button("🔄 Fetch Live Predictions", type="primary"):
        with st.spinner("Loading game data, computing features, fetching odds..."):
            predictions, remaining = get_predictions()
            st.session_state["predictions"] = predictions
            st.session_state["remaining"] = remaining

    if "predictions" in st.session_state:
        predictions = st.session_state["predictions"]
        remaining = st.session_state.get("remaining", "?")

        st.info(f"📡 {len(predictions)} games found • API requests remaining: {remaining}")

        for p in predictions:
            pst = timezone(timedelta(hours=-7))
            local_time = p["commence_time"].astimezone(pst)
            time_str = local_time.strftime("%b %d, %I:%M %p PT")

            with st.container():
                st.markdown(f"### {p['away_team']} @ {p['home_team']}")
                st.caption(time_str)

                col1, col2, col3 = st.columns(3)

                # Moneyline
                with col1:
                    st.markdown("**Moneyline**")
                    prob_pct = p["home_win_prob_calibrated"] * 100
                    st.metric(
                        label=f"{p['home_abbr']} Win Probability",
                        value=f"{prob_pct:.1f}%",
                    )
                    if p["home_ml"]:
                        st.caption(f"Market: {p['home_abbr']} {p['home_ml']:+d} | {p['away_abbr']} {p['away_ml']:+d}")
                    if p["ml_edge"]:
                        edge_pct = p["ml_edge"] * 100
                        st.caption(f"Edge: {edge_pct:+.1f}%")

                    if "Bet" in p["ml_recommendation"]:
                        st.success(f"✅ {p['ml_recommendation']}")
                    else:
                        st.caption(f"🚫 {p['ml_recommendation']}")

                # Spread
                with col2:
                    st.markdown("**Spread**")
                    st.metric(
                        label=f"{p['home_abbr']} Predicted Margin",
                        value=f"{p['predicted_margin']:+.1f}",
                    )
                    if p["book_spread"] is not None:
                        st.caption(f"Line: {p['home_abbr']} {p['book_spread']:+.1f}")
                    if p["spread_edge"] is not None:
                        st.caption(f"Edge: {p['spread_edge']:+.1f} points")

                    if "Bet" in p["spread_recommendation"]:
                        st.success(f"✅ {p['spread_recommendation']}")
                    else:
                        st.caption(f"🚫 {p['spread_recommendation']}")

                # Totals
                with col3:
                    st.markdown("**Totals**")
                    st.metric(
                        label="Predicted Total",
                        value=f"{p['predicted_total']:.1f}",
                    )
                    if p["book_total"] is not None:
                        st.caption(f"Line: O/U {p['book_total']}")
                    if p["totals_edge"] is not None:
                        direction = "OVER" if p["totals_edge"] > 0 else "UNDER"
                        st.caption(f"Edge: {abs(p['totals_edge']):.1f} pts {direction}")

                    if "Bet" in p["totals_recommendation"]:
                        st.success(f"✅ {p['totals_recommendation']}")
                    else:
                        st.caption(f"🚫 {p['totals_recommendation']}")

                st.divider()

        # Summary
        ml_bets = [p for p in predictions if "Bet" in p["ml_recommendation"]]
        spread_bets = [p for p in predictions if "Bet" in p["spread_recommendation"]]
        totals_bets = [p for p in predictions if "Bet" in p["totals_recommendation"]]

        st.subheader("Bet Recommendations Summary")
        scol1, scol2, scol3 = st.columns(3)
        scol1.metric("Moneyline Bets", len(ml_bets))
        scol2.metric("Spread Bets", len(spread_bets))
        scol3.metric("Totals Bets", len(totals_bets))

    else:
        st.info("Click **Fetch Live Predictions** to load today's games and odds.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════

elif page == "Model Comparison":
    st.title("Model Comparison")
    st.caption("Performance of Logistic Regression, Random Forest, and XGBoost across all bet types")

    params = load_strategy_params()

    # Moneyline Results
    st.subheader("Moneyline Models (Classification)")

    ml_data = {
        "Model": ["Logistic Regression", "Random Forest ⭐", "XGBoost", "Baseline (always home)"],
        "Accuracy": ["69.7%", "71.0%", "69.5%", "54.1%"],
        "AUC": ["0.7451", "0.7659", "0.7526", "—"],
        "ECE (raw)": ["0.0412", "0.0519", "0.0214", "—"],
        "ECE (calibrated)": ["0.0341", "0.0293", "0.0209", "—"],
        "Best Calibration": ["Temperature", "Platt", "Platt", "—"],
    }
    st.dataframe(pd.DataFrame(ml_data), use_container_width=True, hide_index=True)

    st.info("**Key finding:** Random Forest achieves the highest accuracy (71.0%) and AUC (0.7659). "
            "XGBoost has the best calibration (ECE 0.0209). All models significantly outperform the 54.1% baseline.")

    # Spread Results
    st.subheader("Spread Models (Regression)")

    spread_data = {
        "Model": ["Ridge Regression", "Random Forest ⭐", "XGBoost"],
        "MAE": ["11.14", "10.91", "11.07"],
        "RMSE": ["14.16", "13.83", "14.08"],
        "R²": ["0.2137", "0.2507", "0.2230"],
    }
    st.dataframe(pd.DataFrame(spread_data), use_container_width=True, hide_index=True)

    st.info("**Key finding:** Random Forest explains 25% of point differential variance (R² = 0.2507). "
            "The combined strategy with ML confidence filter achieved 58.1% win rate and +10.96% ROI in backtesting.")

    # Totals Results
    st.subheader("Totals Models (Regression)")

    totals_data = {
        "Model": ["Ridge Regression", "Random Forest", "XGBoost"],
        "MAE": ["14.75", "14.87", "14.97"],
        "R²": ["0.1371", "0.1310", "0.1176"],
        "Win Rate": ["50.3%", "48.8%", "50.2%"],
        "ROI": ["-3.95%", "-6.86%", "-4.25%"],
    }
    st.dataframe(pd.DataFrame(totals_data), use_container_width=True, hide_index=True)

    st.warning("**Key finding:** No totals model achieves profitability. The over/under market "
               "appears more efficiently priced than moneylines or spreads. This is a valid negative "
               "result — the market is too efficient for team-level features alone.")

    # Feature Engineering Impact
    st.subheader("Feature Engineering Impact")

    feature_data = {
        "Feature Set": [
            "Original (rolling stats)",
            "Enhanced (+ spread/totals features)",
            "Advanced (+ Elo + Four Factors)",
            "Final (+ L20/L30 windows)",
        ],
        "Features": [70, 176, 219, 297],
        "RF Accuracy": ["65.4%", "65.3%", "67.6%", "70.5%"],
        "RF AUC": ["0.7014", "0.6961", "0.7286", "0.7629"],
        "Spread R²": ["0.1580", "0.1596", "0.2507", "—"],
    }
    st.dataframe(pd.DataFrame(feature_data), use_container_width=True, hide_index=True)

    st.info("**Key finding:** Feature engineering had a larger impact than model selection. "
            "Elo ratings and Four Factors alone improved accuracy from 65.4% to 67.6%. "
            "Adding L20/L30 windows pushed it to 70.5%.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3: BACKTESTING
# ══════════════════════════════════════════════════════════════════════════

elif page == "Backtesting":
    st.title("Backtesting Results")
    st.caption("Historical performance against real sportsbook odds (2024-25 test season)")

    params = load_strategy_params()

    # Moneyline Backtesting
    st.subheader("Moneyline Backtesting")

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("ROI", "+83.25%")
    mcol2.metric("Win Rate", "57.9%")
    mcol3.metric("Total Bets", "302")
    mcol4.metric("Profit", "+$2,514")

    st.caption("Strategy: Calibrated Random Forest, 3% minimum edge, $10 flat bets against real Kaggle odds")

    st.divider()

    # Spread Backtesting
    st.subheader("Spread Backtesting (Combined Strategy)")

    scol1, scol2, scol3, scol4 = st.columns(4)
    scol1.metric("ROI", "+10.96%")
    scol2.metric("Win Rate", "58.1%")
    scol3.metric("Total Bets", "160")
    scol4.metric("Profit", "+$175")

    st.caption("Strategy: Spread edge ≥ 4.0 pts + ML confidence ≥ 75%, $10 flat bets at -110 odds")

    # Combined strategy table
    st.subheader("Spread Strategy Parameter Sweep")

    sweep_data = {
        "Edge Threshold": ["3.0", "4.0", "4.0", "4.0", "5.0"],
        "ML Confidence": ["0.75", "0.65", "0.73", "0.75", "0.75"],
        "Bets": [162, 376, 215, 160, 152],
        "Win Rate": ["58.0%", "54.3%", "56.3%", "58.1%", "57.9%"],
        "ROI": ["+10.77%", "+3.57%", "+7.44%", "+10.96%", "+10.52%"],
        "Profit": ["+$174", "+$134", "+$160", "+$175", "+$160"],
    }
    st.dataframe(pd.DataFrame(sweep_data), use_container_width=True, hide_index=True)

    st.info("**Key finding:** Higher ML confidence thresholds produce better win rates and ROI "
            "but fewer bets. The optimal balance is Edge ≥ 4.0 + Confidence ≥ 0.75: "
            "58.1% win rate on 160 bets for +10.96% ROI.")

    st.divider()

    # Totals
    st.subheader("Totals Backtesting")
    st.warning("Totals prediction was not profitable across any model or threshold combination. "
               "Best win rate: 50.3% (below 52.4% break-even). "
               "This suggests the over/under market is more efficiently priced than moneylines or spreads.")

    st.divider()

    # Calibration Impact
    st.subheader("Calibration vs Accuracy: The Core Thesis")

    ccol1, ccol2 = st.columns(2)
    with ccol1:
        st.markdown("**Calibrated Model**")
        st.metric("ROI", "+83.25%")
        st.metric("Win Rate", "57.9%")
        st.metric("Bets Placed", "302")

    with ccol2:
        st.markdown("**Uncalibrated Model**")
        st.metric("ROI", "+79.17%")
        st.metric("Win Rate", "53.2%")
        st.metric("Bets Placed", "310")

    st.info("**Key finding:** Calibrated models achieve higher ROI with fewer bets and a higher win rate. "
            "This confirms Walsh & Joshi (2024): calibration matters more than accuracy for profitability.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4: BET TRACKER
# ══════════════════════════════════════════════════════════════════════════

elif page == "Bet Tracker":
    st.title("Personal Bet Tracker")
    st.caption("Track real bets placed based on model recommendations")

    df = load_bet_tracker()

    if len(df) == 0:
        st.info("No bets logged yet. Add bets to `data/bet_tracker.csv` to see them here.")
    else:
        # Summary metrics
        resolved = df[df["result"].isin(["win", "loss", "push"])] if "result" in df.columns else pd.DataFrame()
        pending = df[df["result"] == "pending"] if "result" in df.columns else df

        if len(resolved) > 0:
            wins = len(resolved[resolved["result"] == "win"])
            losses = len(resolved[resolved["result"] == "loss"])
            total_profit = resolved["units_profit"].astype(float).sum() if "units_profit" in resolved.columns else 0
            total_wagered = resolved["units_wagered"].astype(float).sum() if "units_wagered" in resolved.columns else 0
            win_rate = wins / len(resolved) * 100

            bcol1, bcol2, bcol3, bcol4 = st.columns(4)
            bcol1.metric("Record", f"{wins}W - {losses}L")
            bcol2.metric("Win Rate", f"{win_rate:.1f}%")
            bcol3.metric("Units Profit", f"{total_profit:+.2f}")
            bcol4.metric("ROI", f"{(total_profit / total_wagered * 100):+.1f}%" if total_wagered > 0 else "—")

            # By bet type
            st.subheader("By Bet Type")
            for bet_type in ["moneyline", "spread", "totals"]:
                type_bets = resolved[resolved["bet_type"] == bet_type] if "bet_type" in resolved.columns else pd.DataFrame()
                if len(type_bets) > 0:
                    type_wins = len(type_bets[type_bets["result"] == "win"])
                    type_profit = type_bets["units_profit"].astype(float).sum()
                    type_wr = type_wins / len(type_bets) * 100
                    st.caption(f"**{bet_type.upper()}:** {type_wins}/{len(type_bets)} ({type_wr:.1f}%) | Profit: {type_profit:+.2f}u")

        if len(pending) > 0:
            st.subheader(f"Pending Bets ({len(pending)})")
            st.dataframe(pending, use_container_width=True, hide_index=True)

        if len(resolved) > 0:
            st.subheader(f"Resolved Bets ({len(resolved)})")
            st.dataframe(resolved, use_container_width=True, hide_index=True)

        # Full table
        st.subheader("All Bets")
        st.dataframe(df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 5: ABOUT
# ══════════════════════════════════════════════════════════════════════════

elif page == "About":
    st.title("About This Project")

    st.markdown("""
    ### National Databall Association
    **Machine Learning Techniques for Sports Betting Prediction System**

    *CPSC 597: Project Seminar — California State University, Fullerton*
    *Aaron Tang | Supervisor: Dr. Duy H. Ho | May 2026*

    ---

    ### Core Thesis
    Calibration-optimized machine learning models produce higher betting ROI than
    accuracy-optimized models. A model that says a team has a 70% chance of winning
    should win roughly 70% of the time — when this holds, bettors can size wagers
    optimally using the Kelly Criterion.

    ### System Architecture
    - **Data Collection:** 8,784 games across 7 NBA seasons (2019-2025) from Basketball Reference and NBA API
    - **Feature Engineering:** 297 features including rolling team stats, Elo ratings, and Dean Oliver's Four Factors
    - **Models:** Logistic Regression, Random Forest, XGBoost for each of 3 bet types
    - **Calibration:** Platt scaling, isotonic regression, temperature scaling
    - **Live Odds:** The Odds API for real-time moneyline, spread, and totals from DraftKings and FanDuel

    ### Key Results
    | Bet Type | Best Model | Accuracy/Win Rate | ROI |
    |----------|-----------|-------------------|-----|
    | Moneyline | Random Forest | 71.0% accuracy | +83.25% |
    | Spread | RF Combined | 58.1% win rate | +10.96% |
    | Totals | — | Below break-even | Not profitable |

    ### Tech Stack
    Python, scikit-learn, XGBoost, pandas, Streamlit, The Odds API

    ### References
    - Walsh & Joshi (2024) — Calibration vs accuracy for sports betting
    - Dean Oliver — Basketball on Paper: Four Factors of basketball success
    - Beal et al. (2020) — DFS optimization with ML
    """)