"""
Data Fetching & Preprocessing Module
======================================
Fetches historical + live stock data using yfinance.
Applies normalization, feature extraction, and sliding-window construction.

WHAT THIS MODULE DOES:
1. StockDataFetcher  -> downloads OHLCV data from Yahoo Finance via yfinance
2. add_technical_features -> computes RSI, MACD, Bollinger Bands, Moving Averages etc.
3. DataPreprocessor  -> scales data (MinMaxScaler), builds sliding windows for ANN input,
                        splits data chronologically into Train / Val / Test sets

WHY SLIDING WINDOW?
The ANN needs to see PATTERNS over time, not just one day's price.
A window of 30 days means: "given the last 30 days, predict tomorrow."
Each window is flattened into a 1D vector and fed to the ANN's input layer.

WHY MINMAX SCALING?
Neural networks learn faster and more stably when all inputs are in [0, 1].
Raw prices (e.g. 150 to 200) and volume (e.g. 50,000,000) are on very different
scales -- without scaling the model would be confused by the magnitude differences.

WHY CHRONOLOGICAL SPLIT?
Financial data is time-ordered. If we shuffle randomly, the model could
"see the future" during training (data leakage). So we always split in order:
  oldest 70% -> train
  next 15%   -> validation
  newest 15% -> test (final unseen evaluation)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)


# -- Technical Indicator Helpers -----------------------------------------------

def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI).
    Measures overbought (>70) or oversold (<30) conditions.
    Range: 0 to 100.
    """
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _compute_macd(series: pd.Series) -> pd.DataFrame:
    """
    MACD (Moving Average Convergence Divergence).
    Identifies trend direction and momentum.
    MACD line = 12-day EMA - 26-day EMA
    Signal line = 9-day EMA of MACD
    """
    ema12  = series.ewm(span=12, adjust=False).mean()
    ema26  = series.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return pd.DataFrame({"MACD": macd, "MACD_Signal": signal})


def _compute_bollinger(series: pd.Series, window: int = 20) -> pd.DataFrame:
    """
    Bollinger Bands.
    Upper = 20-day SMA + 2*std, Lower = 20-day SMA - 2*std.
    When price is near upper band: overbought signal.
    When price is near lower band: oversold signal.
    """
    sma   = series.rolling(window).mean()
    std   = series.rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    width = (upper - lower) / (sma + 1e-9)
    return pd.DataFrame({"BB_Upper": upper, "BB_Lower": lower, "BB_Width": width})


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 14 technical indicators to the OHLCV dataframe.
    These give the ANN richer signals beyond just raw price.

    Features added:
        MA5, MA10, MA20, MA50    : Moving averages (trend smoothing)
        RSI                      : Momentum / overbought-oversold
        MACD, MACD_Signal        : Trend direction + momentum
        BB_Upper, BB_Lower, BB_Width : Volatility bands
        Log_Return               : Daily percentage change (log scale)
        Volatility               : Rolling 10-day std of log returns
        PV_Trend                 : Price x Volume cumulative trend
        Close_MA20_Ratio         : How far price is from its 20-day average
    """
    close  = df["Close"]
    volume = df["Volume"]
    df     = df.copy()

    for w in [5, 10, 20, 50]:
        df[f"MA{w}"] = close.rolling(w).mean()

    df["RSI"] = _compute_rsi(close)

    macd_df           = _compute_macd(close)
    df["MACD"]        = macd_df["MACD"]
    df["MACD_Signal"] = macd_df["MACD_Signal"]

    bb_df          = _compute_bollinger(close)
    df["BB_Upper"] = bb_df["BB_Upper"]
    df["BB_Lower"] = bb_df["BB_Lower"]
    df["BB_Width"] = bb_df["BB_Width"]

    df["Log_Return"]        = np.log(close / close.shift(1))
    df["Volatility"]        = df["Log_Return"].rolling(10).std()
    df["PV_Trend"]          = (close.pct_change() * volume).cumsum()
    df["Close_MA20_Ratio"]  = close / (df["MA20"] + 1e-9)

    return df.dropna()


# -- Data Fetcher --------------------------------------------------------------

class StockDataFetcher:
    """Downloads stock data from Yahoo Finance using yfinance."""

    def __init__(self, ticker: str, period: str = "5y", interval: str = "1d"):
        self.ticker   = ticker.upper()
        self.period   = period
        self.interval = interval
        self.raw_df   = None

    def fetch(self) -> pd.DataFrame:
        """Fetch full historical OHLCV data and add technical features."""
        logger.info("Downloading %s | period=%s | interval=%s",
                    self.ticker, self.period, self.interval)
        try:
            df = yf.download(
                self.ticker,
                period=self.period,
                interval=self.interval,
                auto_adjust=True,
                progress=False,
            )
        except Exception as e:
            raise RuntimeError(f"yfinance download failed: {e}") from e

        if df.empty:
            raise ValueError(f"No data returned for ticker '{self.ticker}'. "
                             "Check the ticker symbol.")

        # Flatten MultiIndex columns (yfinance sometimes returns them)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        df = add_technical_features(df)
        self.raw_df = df
        logger.info("Downloaded %d rows with %d features.", len(df), df.shape[1])
        return df

    def fetch_live(self, lookback_days: int = 60) -> pd.DataFrame:
        """
        Fetch the most recent N days of data for live prediction.
        Downloads extra buffer days so technical indicators can be computed.
        """
        logger.info("Fetching live data for %s...", self.ticker)
        end   = datetime.today()
        start = end - timedelta(days=lookback_days + 70)   # extra buffer for indicators
        try:
            df = yf.download(
                self.ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval=self.interval,
                auto_adjust=True,
                progress=False,
            )
        except Exception as e:
            raise RuntimeError(f"Live data fetch failed: {e}") from e

        if df.empty:
            raise ValueError("No live data returned. Market may be closed.")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        df = add_technical_features(df)
        return df.tail(lookback_days)


# -- Preprocessor -------------------------------------------------------------

class DataPreprocessor:
    """
    Scales data and builds sliding-window input sequences for the ANN.

    Sliding window logic:
        If window_size = 30 and we have 1000 days of data:
        - Sample 1: days 0-29  -> predict day 30
        - Sample 2: days 1-30  -> predict day 31
        - ...
        - Sample 970: days 970-999 -> predict day 1000

    Each window is flattened: (30 days x 19 features) = 570 inputs to ANN.
    """

    def __init__(self, window_size: int = 30, test_size: float = 0.15,
                 val_size: float = 0.15):
        self.window_size    = window_size
        self.test_size      = test_size
        self.val_size       = val_size
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler  = MinMaxScaler(feature_range=(0, 1))
        self.feature_cols: List[str] = []

    def _build_sequences(self, features: np.ndarray,
                         target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(self.window_size, len(features)):
            window = features[i - self.window_size: i]  # shape: (window_size, n_features)
            X.append(window.flatten())                   # flatten to 1D for Dense ANN
            y.append(target[i])
        return np.array(X), np.array(y)

    def prepare(self, df: pd.DataFrame) -> Tuple:
        """
        Full preprocessing pipeline.
        Returns: X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Target = Close price
        target = df["Close"].values.reshape(-1, 1)

        # Features = everything except Close
        self.feature_cols = [c for c in df.columns if c != "Close"]
        features = df[self.feature_cols].values

        # Scale to [0, 1]
        features_scaled = self.feature_scaler.fit_transform(features)
        target_scaled   = self.target_scaler.fit_transform(target).flatten()

        X, y = self._build_sequences(features_scaled, target_scaled)

        # Chronological split -- no shuffling!
        n      = len(X)
        n_test = int(n * self.test_size)
        n_val  = int((n - n_test) * self.val_size)

        X_test,  y_test  = X[-n_test:],         y[-n_test:]
        X_rem,   y_rem   = X[:-n_test],         y[:-n_test]
        X_val,   y_val   = X_rem[-n_val:],      y_rem[-n_val:]
        X_train, y_train = X_rem[:-n_val],      y_rem[:-n_val]

        logger.info("Split -> Train: %d | Val: %d | Test: %d",
                    len(X_train), len(X_val), len(X_test))
        return X_train, X_val, X_test, y_train, y_val, y_test

    def prepare_live(self, live_df: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare the latest window for live inference."""
        if not self.feature_cols:
            logger.error("Preprocessor not fitted. Run prepare() first.")
            return None
        df = live_df[self.feature_cols]
        if len(df) < self.window_size:
            logger.warning("Not enough live data rows (%d) for window size (%d).",
                           len(df), self.window_size)
            return None
        window        = df.values[-self.window_size:]
        window_scaled = self.feature_scaler.transform(window)
        return window_scaled.flatten().reshape(1, -1)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "preprocessor.pkl"), "wb") as f:
            pickle.dump(self, f)
        logger.info("Preprocessor saved to %s", path)

    @staticmethod
    def load(path: str) -> "DataPreprocessor":
        with open(os.path.join(path, "preprocessor.pkl"), "rb") as f:
            obj = pickle.load(f)
        logger.info("Preprocessor loaded from %s", path)
        return obj