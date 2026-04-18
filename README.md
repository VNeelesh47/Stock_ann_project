# Stock Market Price Prediction Using ANN
### Major Project | Python · TensorFlow/Keras · yfinance

---

## Overview

This project implements a **Multi-layer Feedforward Artificial Neural Network (ANN)** trained with the **Backpropagation** algorithm to predict stock market prices. It fetches **live and historical data** from Yahoo Finance using the `yfinance` library, performs feature engineering, trains the ANN, evaluates using standard financial metrics, and provides **near real-time predictions** through a continuous live update system.

---

## Project Structure

```
stock_ann_project/
├── main.py                      # Orchestrator — run this
├── requirements.txt
│
├── models/
│   ├── __init__.py
│   ├── ann_model.py             # ANN architecture + training + evaluation
│   └── saved/                   # Auto-created: saved weights + config
│
├── utils/
│   ├── __init__.py
│   ├── data_utils.py            # yfinance fetcher + feature engineering + preprocessor
│   ├── visualizer.py            # Charts (Matplotlib) + HTML report generator
│   └── live_predictor.py        # Background thread for live inference
│
├── reports/                     # Auto-created: PNGs + HTML report per ticker
├── logs/                        # pipeline.log
└── README.md
```

---

## Setup & Installation

```bash
# 1. Clone / unzip the project folder
cd stock_ann_project

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Full Pipeline (Fetch → Train → Evaluate → Report)

```bash
python main.py --ticker AAPL
python main.py --ticker RELIANCE.NS --period 3y --window 20
python main.py --ticker TSLA --epochs 200 --hidden-layers 256 128 64 32
```

### Live-Only Mode (load saved model, run continuous predictions)

```bash
python main.py --ticker AAPL --live-only
python main.py --ticker AAPL --live-only --live-interval 60   # update every 60s
```

### All Arguments

| Argument | Default | Description |
|---|---|---|
| `--ticker` | `AAPL` | Yahoo Finance ticker (e.g. `RELIANCE.NS`, `^NSEI`) |
| `--period` | `5y` | Historical data period (`1y`, `2y`, `5y`, `10y`, `max`) |
| `--interval` | `1d` | OHLCV interval (`1d`, `1h`, `5m`) |
| `--window` | `30` | Sliding window size (days fed to ANN) |
| `--hidden-layers` | `128 64 32` | Neurons per hidden layer |
| `--lr` | `0.001` | Learning rate (Adam optimizer) |
| `--epochs` | `150` | Maximum training epochs |
| `--batch-size` | `32` | Mini-batch size |
| `--dropout` | `0.2` | Dropout rate per hidden layer |
| `--live-only` | `False` | Skip training; load saved model |
| `--live-interval` | `300` | Seconds between live predictions |
| `--live-duration` | `0` | Seconds to run live loop (0 = indefinite) |
| `--output-dir` | `reports` | Chart + HTML report destination |
| `--model-dir` | `models/saved` | Model save/load path |

---

## Architecture

```
Input Layer  →  Hidden Layer 1 (128, ReLU + BN + Dropout)
             →  Hidden Layer 2 (64, ReLU + BN + Dropout)
             →  Hidden Layer 3 (32, ReLU + BN + Dropout)
             →  Output Layer   (1, Linear)

Loss    : Huber Loss (robust to price outliers)
Optimizer: Adam with ReduceLROnPlateau
Training : Backpropagation with Early Stopping
```

---

## Features Used as ANN Input

Each time-step in the sliding window includes:
- **OHLCV**: Open, High, Low, Close, Volume
- **Moving Averages**: MA5, MA10, MA20, MA50
- **RSI** (14-period)
- **MACD** + MACD Signal
- **Bollinger Band** upper/lower/width
- **Log Returns** + Rolling Volatility
- **Price-Volume Trend**
- **Close/MA20 Ratio**

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **MAPE (%)** | Mean Absolute Percentage Error |
| **RMSE** | Root Mean Squared Error |
| **MAE** | Mean Absolute Error |
| **Directional Accuracy (%)** | % of up/down moves correctly predicted |

---

## Output Files

After running, find in `reports/<TICKER>/`:
- `<TICKER>_report.html` — Full HTML report (open in browser)
- `<TICKER>_predictions.png` — Actual vs predicted + residuals
- `training_history.png` — Loss & MAE curves
- `error_distribution.png` — Histogram of prediction errors
- `scatter_actual_vs_pred.png` — Scatter of actual vs predicted

---

## Technologies

| Library | Purpose |
|---|---|
| `yfinance` | Live & historical stock data from Yahoo Finance |
| `TensorFlow / Keras` | ANN model (feedforward + backpropagation) |
| `scikit-learn` | MinMaxScaler, train/test split, MAPE/RMSE |
| `pandas` / `numpy` | Data manipulation & feature engineering |
| `matplotlib` | Charts and visualizations |

---

## Notes

- Data is split **chronologically** (no shuffling) to respect time-series structure.
- The model is retrained from scratch each run; `--live-only` loads saved weights.
- Indian NSE stocks: use `.NS` suffix (e.g. `TCS.NS`, `INFY.NS`).
- Live predictions run in a **daemon thread** — safe to Ctrl+C at any time.

---

*Stock Market Price Prediction Using Artificial Neural Network on Live Data from yfinance*  
*Major Project — Python · TensorFlow/Keras · yfinance*
