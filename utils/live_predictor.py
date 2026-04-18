"""
Live Prediction Engine
Periodically fetches fresh data from yfinance and updates price forecasts.
Runs as a background thread — no external broker/server required.
"""

import time
import threading
import logging
from datetime import datetime
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class LivePredictor:
    """
    Continuously fetches live stock data and generates updated ANN predictions.

    Usage:
        predictor = LivePredictor(fetcher, preprocessor, model, ticker)
        predictor.start(interval_seconds=300)  # update every 5 minutes
        ...
        predictor.stop()
    """

    def __init__(self, fetcher, preprocessor, model, ticker: str,
                 on_prediction: Optional[Callable] = None):
        """
        Args:
            fetcher        : StockDataFetcher instance (configured for live)
            preprocessor   : Fitted DataPreprocessor instance
            model          : Trained StockANN instance
            ticker         : Stock ticker symbol
            on_prediction  : Optional callback(result_dict) called after each update
        """
        self.fetcher       = fetcher
        self.preprocessor  = preprocessor
        self.model         = model
        self.ticker        = ticker
        self.on_prediction = on_prediction

        self._thread: Optional[threading.Thread] = None
        self._stop_event   = threading.Event()
        self.latest_result: Optional[dict] = None

    # ------------------------------------------------------------------
    def _predict_once(self) -> Optional[dict]:
        """Fetch live data, prepare window, run inference."""
        try:
            live_df = self.fetcher.fetch_live(lookback_days=60)
        except Exception as e:
            logger.error("Live data fetch error: %s", e)
            return None

        X_live = self.preprocessor.prepare_live(live_df)
        if X_live is None:
            logger.warning("Insufficient live data for prediction window.")
            return None

        y_scaled = self.model.predict(X_live)
        y_price  = self.preprocessor.target_scaler.inverse_transform(
            y_scaled.reshape(-1, 1)
        ).flatten()[0]

        last_close = float(live_df["Close"].iloc[-1])
        change_pct = ((y_price - last_close) / last_close) * 100

        result = {
            "ticker":          self.ticker,
            "timestamp":       datetime.now().isoformat(),
            "last_close":      round(last_close, 4),
            "predicted_price": round(y_price, 4),
            "change_pct":      round(change_pct, 3),
            "direction": "▲ UP" if change_pct >= 0 else "▼ DOWN",
        }

        self.latest_result = result
        logger.info(
            "[LIVE] %s | Last Close: %.4f | Predicted: %.4f | Change: %+.3f%%",
            self.ticker, last_close, y_price, change_pct,
        )

        if self.on_prediction:
            try:
                self.on_prediction(result)
            except Exception as cb_err:
                logger.error("Prediction callback error: %s", cb_err)

        return result

    # ------------------------------------------------------------------
    def _run_loop(self, interval_seconds: int):
        """Internal loop executed in a background thread."""
        logger.info("Live prediction loop started (interval: %ds).", interval_seconds)
        while not self._stop_event.is_set():
            self._predict_once()
            self._stop_event.wait(timeout=interval_seconds)
        logger.info("Live prediction loop stopped.")

    # ------------------------------------------------------------------
    def start(self, interval_seconds: int = 300):
        """Start background prediction loop."""
        if self._thread and self._thread.is_alive():
            logger.warning("LivePredictor already running.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(interval_seconds,),
            daemon=True,
            name="LivePredictorThread",
        )
        self._thread.start()
        logger.info("LivePredictor started for %s.", self.ticker)

    def stop(self):
        """Gracefully stop the prediction loop."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("LivePredictor stopped.")

    def predict_now(self) -> Optional[dict]:
        """Run a single prediction immediately (blocking)."""
        return self._predict_once()
