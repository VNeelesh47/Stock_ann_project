# Stock Market Price Prediction Using ANN
### Major Project | Python + TensorFlow/Keras + yfinance

---

## Project Overview

This project implements a **Multi-layer Feedforward Artificial Neural Network (ANN)**
trained with **Backpropagation** to predict stock market prices.

It fetches **live and historical data** from Yahoo Finance using `yfinance`,
performs feature engineering, trains the ANN, evaluates using standard financial
metrics, and provides **continuous live predictions** through an automated update system.

---

## Project Structure

```
Stock_ann_project/
|
|-- main.py                  <- Run this. Has interactive menu + command-line support
|-- requirements.txt         <- All dependencies
|-- README.md
|
|-- models/
|   |-- __init__.py
|   |-- ann_model.py         <- ANN architecture, training, evaluation
|
|-- utils/
|   |-- __init__.py
|   |-- data_utils.py        <- yfinance fetcher + feature engineering + preprocessor
|   |-- visualizer.py        <- Matplotlib charts + HTML report generator
|   |-- live_predictor.py    <- Continuous live prediction engine (background thread)
|
|-- logs/                    <- Auto-created: pipeline.log + live prediction logs
|-- reports/                 <- Auto-created: charts + HTML report per ticker
|-- models/saved/            <- Auto-created: saved model weights
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/VNeelesh47/Stock_ann_project.git
cd Stock_ann_project

# 2. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Option 1 -- Interactive Menu (easiest)
Just run with no arguments and choose from the menu:

```bash
python main.py
```

You will see:

```
============================================================
   STOCK MARKET PRICE PREDICTION USING ANN
   Major Project | TensorFlow/Keras + yfinance
============================================================

What would you like to do?

  [1]  Train model on a stock and generate full report
  [2]  Predict next price  (load saved model, single prediction)
  [3]  Live prediction loop (continuous updates every N seconds)
  [4]  Exit
```

### Option 2 -- Command Line (direct)

```bash
# Train on Apple stock (5 years of data)
python main.py --ticker AAPL

# Train on Indian stock (Reliance, NSE)
python main.py --ticker RELIANCE.NS --period 3y

# Train with custom settings
python main.py --ticker TSLA --epochs 200 --hidden-layers 256 128 64 32

# Load saved model, run live prediction loop
python main.py --ticker AAPL --live-only

# Live loop with custom interval (every 60 seconds)
python main.py --ticker AAPL --live-only --live-interval 60
```

---

## All Command Line Arguments

| Argument | Default | Description |
|---|---|---|
| `--ticker` | AAPL | Yahoo Finance ticker (RELIANCE.NS, TCS.NS, ^NSEI etc.) |
| `--period` | 5y | Historical data period (1y, 2y, 5y, 10y, max) |
| `--interval` | 1d | Data interval (1d, 1h, 5m) |
| `--window` | 30 | Sliding window size in days |
| `--hidden-layers` | 128 64 32 | Neurons per hidden layer |
| `--lr` | 0.001 | Learning rate |
| `--epochs` | 150 | Maximum training epochs |
| `--batch-size` | 32 | Mini-batch size |
| `--dropout` | 0.2 | Dropout rate |
| `--live-only` | False | Skip training, load saved model, run live loop |
| `--live-interval` | 300 | Seconds between live predictions |
| `--live-duration` | 0 | How long to run live loop in seconds (0 = indefinite) |
| `--output-dir` | reports | Folder for charts and HTML report |
| `--model-dir` | models/saved | Folder to save/load model |

---

## ANN Architecture

```
Input Layer
  -> 30 days x 19 features = 570 inputs

Hidden Layer 1  ->  128 neurons (ReLU + BatchNorm + Dropout 20%)
Hidden Layer 2  ->  64  neurons (ReLU + BatchNorm + Dropout 20%)
Hidden Layer 3  ->  32  neurons (ReLU + BatchNorm + Dropout 20%)

Output Layer    ->  1 neuron (Linear) = predicted price
```

**Why ANN?**
Stock prices follow non-linear patterns that linear models cannot capture.
ANN learns these hidden patterns through backpropagation across thousands
of training examples.

**Why Huber Loss?**
Stock prices have sudden spikes. MSE penalizes spikes too heavily and
destabilizes training. Huber Loss behaves like MSE for small errors
and like MAE for large errors -- much more stable for financial data.

**Why Adam Optimizer?**
Adam uses adaptive learning rates per parameter. Works better than
plain SGD for financial data which has irregular, non-stationary patterns.

---

## Features Used as ANN Input

Each day in the sliding window includes these 19 features:

| Feature | What it represents |
|---|---|
| Open, High, Low, Volume | Raw OHLCV data |
| MA5, MA10, MA20, MA50 | Moving averages (trend smoothing) |
| RSI | Overbought / oversold signal (0-100) |
| MACD, MACD_Signal | Trend direction and momentum |
| BB_Upper, BB_Lower, BB_Width | Volatility bands |
| Log_Return | Daily price change (log scale) |
| Volatility | Rolling 10-day std of log returns |
| PV_Trend | Price x Volume cumulative trend |
| Close_MA20_Ratio | How far price is from 20-day average |

---

## Evaluation Metrics

After training, the model is tested on the newest 15% of data (never seen during training):

| Metric | What it means |
|---|---|
| MAPE (%) | Average % error. Under 5% is good for stocks |
| RMSE | Average error in price units (e.g. USD) |
| MAE | Mean absolute price deviation |
| R2 Score | 1.0 = perfect fit, 0.0 = no better than mean |
| Directional Accuracy (%) | % of UP/DOWN moves correctly predicted |
| Max Single Error | Worst single prediction error |
| Within 5% Accuracy (%) | % of predictions within 5% of actual price |

Model is also graded automatically:
- MAPE < 3% = EXCELLENT
- MAPE < 5% = GOOD
- MAPE < 10% = ACCEPTABLE
- MAPE > 10% = NEEDS IMPROVEMENT

---

## Output Files

After running, find everything in `reports/TICKER/`:

| File | Description |
|---|---|
| `TICKER_report.html` | Full HTML report (open in browser) |
| `TICKER_predictions.png` | Actual vs Predicted price + residuals |
| `training_history.png` | Loss and MAE curves over epochs |
| `error_distribution.png` | Histogram of prediction errors |
| `scatter_actual_vs_pred.png` | Scatter: actual vs predicted |

Live prediction logs saved to `logs/TICKER_live_predictions.log`

---

## Live Prediction System

The live system runs as a background thread:

```
fetch latest data from yfinance
        |
        v
prepare 30-day sliding window
        |
        v
ANN forward pass (inference)
        |
        v
inverse-scale to get actual price
        |
        v
log result + print to console
        |
        v
wait N seconds
        |
        v
repeat
```

Every prediction is logged with timestamp, last close, predicted price,
change percentage, and direction (UP/DOWN).

---

## Limitations

1. ANN cannot predict sudden price shocks from news or earnings surprises
2. Model should be retrained periodically as market conditions change
3. Predictions are for next trading day only -- longer forecasts lose accuracy
4. Less reliable during extreme volatility periods

---

## Technologies Used

| Library | Purpose |
|---|---|
| yfinance | Live and historical stock data from Yahoo Finance |
| TensorFlow / Keras | ANN model (feedforward + backpropagation) |
| scikit-learn | MinMaxScaler, MAPE, RMSE, R2 metrics |
| pandas / numpy | Data manipulation and feature engineering |
| matplotlib | Charts and visualizations |

---

## Indian Stock Tickers

For NSE stocks add `.NS` suffix:

```bash
python main.py --ticker RELIANCE.NS
python main.py --ticker TCS.NS
python main.py --ticker INFY.NS
python main.py --ticker HDFCBANK.NS
python main.py --ticker WIPRO.NS
```

---

*Stock Market Price Prediction Using Artificial Neural Network (ANN) on Live Data from yfinance*
*Major Project | TensorFlow/Keras + yfinance + scikit-learn + Matplotlib*