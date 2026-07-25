# Pairs Trading Analytics

End-to-end pipeline for identifying, backtesting, and monitoring statistical arbitrage pairs.

## Project layout

```
v1/
|-- data/
|   |-- raw/          # downloaded OHLC / VIX / S&P 500 CSVs as-is
|   +-- processed/    # master merged CSV, cleaned data
|-- notebooks/        # optional Jupyter exploration
|-- src/
|   |-- phase1_data.py          # data ingestion
|   |-- phase2_eda.py           # exploratory analysis
|   |-- phase3_cointegration.py # pair screening (EG + Johansen)
|   |-- phase3_strategy.py      # backtesting engine
|   |-- phase4_sentiment.py     # news sentiment scoring
|   +-- utils.py                # shared constants & paths
|-- outputs/
|   |-- charts/       # saved figures
|   +-- reports/      # exported tables / summary stats
|-- app/
|   +-- app.py        # Streamlit dashboard (Phase 5)
|-- requirements.txt
+-- README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Running the pipeline

```bash
python src/phase1_data.py          # download and build master data
python src/phase2_eda.py           # exploratory analysis
python src/phase3_cointegration.py # cointegration screening
python src/phase3_strategy.py      # pairs trading backtest
```

## Running the dashboard

```bash
streamlit run app/app.py
```
