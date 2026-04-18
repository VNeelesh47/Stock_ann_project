"""
Data Fetching & Preprocessing Module
Fetches historical + live stock data using yfinance
Applies normalization, feature extraction, and sliding-window construction
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature Engineering Helpers
# ---------------------------------------------------------------------------

def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _compute_macd(series: pd.Series) -> pd.DataFrame:
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return pd.DataFrame({"MACD": macd, "MACD_Signal": signal})


def _compute_bollinger(series: pd.Series, window: int = 20) -> pd.DataFrame:
    sma  = series.rolling(window).mean()
    std  = series.rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    bandwidth = (upper - lower) / (sma + 1e-9)
    return pd.DataFrame({"BB_Upper": upper, "BB_Lower": lower, "BB_Width": bandwidth})


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich OHLCV dataframe with technical indicators.
    Features added:
        - Moving Averages: MA5, MA10, MA20, MA50
        - RSI (14)
        - MACD + MACD Signal
        - Bollinger Band width
        - Log returns
        - Volatility (rolling std of log returns)
        - Price-Volume trend
    """
    close = df["Close"]
    volume = df["Volume"]

    df = df.copy()

    # Moving averages
    for w in [5, 10, 20, 50]:
        df[f"MA{w}"] = close.rolling(w).mean()

    # RSI
    df["RSI"] = _compute_rsi(close)

    # MACD
    macd_df = _compute_macd(close)
    df["MACD"]        = macd_df["MACD"]
    df["MACD_Signal"] = macd_df["MACD_Signal"]

    # Bollinger Bands
    bb_df = _compute_bollinger(close)
    df["BB_Upper"] = bb_df["BB_Upper"]
    df["BB_Lower"] = bb_df["BB_Lower"]
    df["BB_Width"] = bb_df["BB_Width"]

    # Log returns & volatility
    df["Log_Return"]  = np.log(close / close.shift(1))
    df["Volatility"]  = df["Log_Return"].rolling(10).std()

    # Price-Volume Trend
    df["PV_Trend"] = (close.pct_change() * volume).cumsum()

    # Ratio: Close vs MA20
    df["Close_MA20_Ratio"] = close / (df["MA20"] + 1e-9)

    return df.dropna()


# ---------------------------------------------------------------------------
# Data Fetcher
# ---------------------------------------------------------------------------

class StockDataFetcher:
    """Fetches and preprocesses stock data from yfinance."""

    VALID_INTERVALS = {"1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"}

    def __init__(self, ticker: str, period: str = "5y", interval: str = "1d"):
        """
        Args:
            ticker   : Yahoo Finance ticker (e.g. 'AAPL', 'RELIANCE.NS')
            period   : Historical period ('1y', '2y', '5y', '10y', 'max')
            interval : Data interval ('1d', '1h', etc.)
        """
        self.ticker   = ticker.upper()
        self.period   = period
        self.interval = interval
        self.raw_df   = None

    def fetch(self) -> pd.DataFrame:
        """Fetch OHLCV data from yfinance, validate, and add features."""
        logger.info("Fetching data for %s | period=%s | interval=%s",
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
            raise ValueError(f"No data returned for ticker '{self.ticker}'.")

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        df = add_technical_features(df)
        self.raw_df = df
        logger.info("Fetched %d rows with %d features.", len(df), df.shape[1])
        return df

    def fetch_live(self, lookback_days: int = 60) -> pd.DataFrame:
        """
        Fetch the most recent `lookback_days` of data for live inference.
        Returns feature-engineered dataframe.
        """
        logger.info("Fetching live data for %s (last %d days)...", self.ticker, lookback_days)
        end   = datetime.today()
        start = end - timedelta(days=lookback_days + 60)   # extra buffer for indicators
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
            raise RuntimeError(f"Live fetch failed: {e}") from e

        if df.empty:
            raise ValueError("No live data returned.")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        df = add_technical_features(df)
        return df.tail(lookback_days)


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class DataPreprocessor:
    """
    Scales features and constructs sliding-window sequences for the ANN.
    The ANN takes a flat vector of `window_size` × `n_features` as input.
    """

    def __init__(self, window_size: int = 30, test_size: float = 0.15,
                 val_size: float = 0.15):
        """
        Args:
            window_size : Number of past timesteps fed to the ANN
            test_size   : Fraction of data for final testing
            val_size    : Fraction of training data for validation
        """
        self.window_size  = window_size
        self.test_size    = test_size
        self.val_size     = val_size
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler  = MinMaxScaler(feature_range=(0, 1))
        self.feature_cols: List[str] = []

    # ---- Sliding-window builder ----------------------------------------
    def _build_sequences(self, feature_arr: np.ndarray,
                         target_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(self.window_size, len(feature_arr)):
            window = feature_arr[i - self.window_size: i]   # (window_size, n_features)
            X.append(window.flatten())                        # flatten → 1-D for ANN
            y.append(target_arr[i])
        return np.array(X), np.array(y)

    # ---- Main pipeline ---------------------------------------------------
    def prepare(self, df: pd.DataFrame) -> Tuple:
        """
        Full preprocessing pipeline.

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Target: next-day Close price
        target = df["Close"].values.reshape(-1, 1)

        # Feature set: all engineered columns
        self.feature_cols = [c for c in df.columns if c != "Close"]
        features = df[self.feature_cols].values

        # Scale features and target independently
        features_scaled = self.feature_scaler.fit_transform(features)
        target_scaled   = self.target_scaler.fit_transform(target).flatten()

        X, y = self._build_sequences(features_scaled, target_scaled)

        # Chronological split (no shuffling — financial time series)
        n = len(X)
        n_test = int(n * self.test_size)
        n_val  = int((n - n_test) * self.val_size)

        X_test, y_test = X[-n_test:], y[-n_test:]
        X_rem,  y_rem  = X[:-n_test], y[:-n_test]
        X_val,  y_val  = X_rem[-n_val:], y_rem[-n_val:]
        X_train, y_train = X_rem[:-n_val], y_rem[:-n_val]

        logger.info("Dataset split — Train: %d | Val: %d | Test: %d",
                    len(X_train), len(X_val), len(X_test))
        return X_train, X_val, X_test, y_train, y_val, y_test

    # ---- Live window for inference --------------------------------------
    def prepare_live(self, live_df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Prepare the most recent window for live prediction.

        Returns:
            Flat feature vector of shape (1, window_size × n_features) or None
        """
        df = live_df[self.feature_cols]
        if len(df) < self.window_size:
            logger.warning("Not enough live data for a full window.")
            return None
        window = df.values[-self.window_size:]
        window_scaled = self.feature_scaler.transform(window)
        return window_scaled.flatten().reshape(1, -1)

    # ---- Persistence ----------------------------------------------------
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
