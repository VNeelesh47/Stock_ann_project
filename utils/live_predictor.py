"""
Live Prediction Engine
=======================
Periodically fetches fresh data from yfinance and updates price forecasts.
Runs as a background thread -- no external broker or server required.

HOW THE LIVE SYSTEM WORKS:
1. A background thread starts and runs in a loop
2. Every N seconds (default 300 = 5 minutes):
   a. Fetch latest stock data from Yahoo Finance (yfinance)
   b. Prepare the most recent 30-day window
   c. Run ANN forward pass (inference) to get predicted price
   d. Log result + call optional callback function
3. Thread runs until stop() is called or Ctrl+C

This satisfies the PDF requirement:
  "Implement a live update system that periodically retrieves recent
   data and adjusts predictions accordingly."

NOTE ON MARKET HOURS:
  Yahoo Finance data updates during market hours (9:30am - 4pm EST for US).
  Outside those hours the "latest" data will be the most recent closing price.
  The system still runs and predicts correctly -- it just uses last close.
"""

import time
import threading
import logging
from datetime import datetime
from typing import Callable, Optional, List

import numpy as np

logger = logging.getLogger(__name__)


class LivePredictor:
    """
    Continuous live prediction system using ANN.

    Usage:
        predictor = LivePredictor(fetcher, preprocessor, model, ticker)
        predictor.start(interval_seconds=300)   # runs in background thread
        ...
        predictor.stop()                         # graceful shutdown
    """

    def __init__(self, fetcher, preprocessor, model, ticker: str,
                 on_prediction: Optional[Callable] = None):
        """
        Args:
            fetcher        : StockDataFetcher (configured for this ticker)
            preprocessor   : Fitted DataPreprocessor instance
            model          : Trained StockANN instance
            ticker         : Stock ticker symbol (e.g. 'AAPL')
            on_prediction  : Optional callback(result_dict) -- called after each prediction
        """
        self.fetcher       = fetcher
        self.preprocessor  = preprocessor
        self.model         = model
        self.ticker        = ticker
        self.on_prediction = on_prediction

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Track prediction history for analysis
        self.prediction_history: List[dict] = []
        self.latest_result: Optional[dict]  = None

    # -- Core prediction logic -------------------------------------------------

    def _predict_once(self) -> Optional[dict]:
        """
        Single prediction cycle:
          1. Fetch live data from yfinance
          2. Prepare sliding window
          3. ANN forward pass
          4. Inverse-scale to get actual price
          5. Return result dict
        """
        try:
            live_df = self.fetcher.fetch_live(lookback_days=60)
        except Exception as e:
            logger.error("Live data fetch error: %s", e)
            return None

        X_live = self.preprocessor.prepare_live(live_df)
        if X_live is None:
            logger.warning("Insufficient data rows for prediction window. Skipping.")
            return None

        # ANN forward pass (inference only, no training)
        y_scaled = self.model.predict(X_live)
        y_price  = self.preprocessor.target_scaler.inverse_transform(
            y_scaled.reshape(-1, 1)
        ).flatten()[0]

        last_close = float(live_df["Close"].iloc[-1])
        change_pct = ((y_price - last_close) / last_close) * 100

        result = {
            "ticker":          self.ticker,
            "timestamp":       datetime.now().isoformat(timespec="seconds"),
            "last_close":      round(last_close, 4),
            "predicted_price": round(float(y_price), 4),
            "change_pct":      round(change_pct, 3),
            "direction":       "UP" if change_pct >= 0 else "DOWN",
        }

        # Keep last 100 predictions in memory for analysis
        self.prediction_history.append(result)
        if len(self.prediction_history) > 100:
            self.prediction_history.pop(0)

        self.latest_result = result

        logger.info("[LIVE] %s | Close=%.4f | Predicted=%.4f | Change=%+.3f%% | %s",
                    self.ticker, last_close, float(y_price),
                    change_pct, result["direction"])

        if self.on_prediction:
            try:
                self.on_prediction(result)
            except Exception as cb_err:
                logger.error("Callback error: %s", cb_err)

        return result

    # -- Background loop (Gap fixed: "No true live system") --------------------

    def _run_loop(self, interval_seconds: int):
        """
        Continuous loop that runs in a background daemon thread.
        fetch -> predict -> wait -> fetch -> predict -> wait -> ...
        """
        logger.info("Live prediction loop started for %s (every %ds).",
                    self.ticker, interval_seconds)
        while not self._stop_event.is_set():
            self._predict_once()
            # Wait for interval OR until stop() is called (whichever comes first)
            self._stop_event.wait(timeout=interval_seconds)
        logger.info("Live prediction loop stopped.")

    def start(self, interval_seconds: int = 300):
        """Start the background prediction loop."""
        if self._thread and self._thread.is_alive():
            logger.warning("LivePredictor already running.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(interval_seconds,),
            daemon=True,                  # thread dies when main program exits
            name=f"LivePredictor-{self.ticker}",
        )
        self._thread.start()
        logger.info("LivePredictor started for %s.", self.ticker)

    def stop(self):
        """Gracefully stop the prediction loop."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=15)
        logger.info("LivePredictor stopped.")

    def predict_now(self) -> Optional[dict]:
        """Run a single prediction immediately (blocking call)."""
        return self._predict_once()

    # -- Prediction history analysis -------------------------------------------

    def get_summary(self) -> dict:
        """
        Summarize all predictions made so far.
        Useful for result interpretation and analysis.
        """
        if not self.prediction_history:
            return {"status": "No predictions made yet."}

        changes = [r["change_pct"] for r in self.prediction_history]
        ups     = sum(1 for r in self.prediction_history if r["direction"] == "UP")
        downs   = len(self.prediction_history) - ups

        return {
            "total_predictions":   len(self.prediction_history),
            "up_signals":          ups,
            "down_signals":        downs,
            "avg_predicted_change": round(float(np.mean(changes)), 3),
            "max_predicted_change": round(float(np.max(changes)), 3),
            "min_predicted_change": round(float(np.min(changes)), 3),
            "last_prediction":     self.latest_result,
        }