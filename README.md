# nationaldataballassociation 🏀

A sports betting and daily fantasy sports prediction system for the NBA using machine learning techniques, with a focus on **calibration-optimized** predictions over raw accuracy.

## Core Thesis

Based on Walsh & Joshi (2024), calibration quality matters far more than accuracy for profitable sports betting. This system optimizes for Expected Calibration Error (ECE) and Brier Score, enabling Kelly Criterion bet sizing that generated 69.86% higher returns than accuracy-optimized approaches.

## Project Structure

```
nationaldataballassociation/
├── app/
│   ├── api/                    # FastAPI endpoints (future)
│   ├── data/
│   │   ├── collectors/         # Data collection from nba_api, Basketball Reference
│   │   └── processors/         # Cleaning, transformation, feature engineering
│   ├── models/                 # Pydantic data models (schemas)
│   ├── ml/
│   │   ├── features/           # Feature engineering pipelines
│   │   ├── models/             # ML model definitions and training
│   │   ├── calibration/        # Calibration techniques (Platt, isotonic, temp scaling)
│   │   └── evaluation/         # Metrics: ECE, Brier, reliability diagrams
│   ├── optimization/           # DFS lineup optimizer (stretch goal)
│   └── utils/                  # Shared utilities, logging, helpers
├── tests/
│   ├── unit/
│   └── integration/
├── notebooks/                  # EDA and experimentation
├── scripts/                    # One-off scripts (data backfill, etc.)
├── config/                     # Environment configs
├── data/                       # Local data cache (gitignored)
├── models/                     # Saved model artifacts (gitignored)
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

## Quick Start

```bash
# Clone the repo
git clone https://github.com/yourusername/nationaldataballassociation.git
cd nationaldataballassociation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -e ".[dev]"

# Copy environment config
cp .env.example .env

# Run data collection (pulls from nba_api)
python -m scripts.collect_seasons --seasons 2019 2020 2021 2022 2023 2024

# Run the pipeline
python -m app.ml.pipeline
```

## Key Concepts

- **Calibration > Accuracy**: A 56% accurate model with good calibration beats a 60% accurate model with poor calibration for betting profitability.
- **Kelly Criterion**: Optimal bet sizing derived from calibrated probability estimates.
- **Reliability Diagrams**: Visual proof that predicted probabilities match observed frequencies.

## Tech Stack

- **Data**: nba_api, Basketball Reference (via requests/BeautifulSoup)
- **ML**: scikit-learn, XGBoost, LightGBM
- **Calibration**: scikit-learn CalibratedClassifierCV, custom ECE implementation
- **Interpretability**: SHAP
- **Backend** (future): FastAPI, PostgreSQL
- **Frontend** : Streamlit

## References

- Walsh & Joshi (2024) — Calibration vs accuracy for sports betting
- Beal et al. (2020) — DFS optimization with ML + mixed-integer programming
- Constantinou & Papastamoulis (2024) — Player-specific modeling for NBA DFS
