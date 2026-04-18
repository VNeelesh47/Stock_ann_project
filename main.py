"""
Stock Market Price Prediction Using ANN — Main Pipeline
==========================================================
Major Project | TensorFlow/Keras + yfinance

Orchestrates:
  1. Live + historical data fetching (yfinance)
  2. Feature engineering & preprocessing
  3. ANN model training (backpropagation)
  4. Evaluation: MAPE, RMSE, MAE, Directional Accuracy
  5. Live prediction updates
  6. HTML + chart report generation

Usage:
    python main.py --ticker AAPL --period 5y --window 30
    python main.py --ticker RELIANCE.NS --period 3y --window 20 --epochs 200
    python main.py --ticker AAPL --live-only   # skip training, run live inference
"""

import argparse
import logging
import os
import sys
import json
import time
import numpy as np

# ── Project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from utils.data_utils      import StockDataFetcher, DataPreprocessor
from utils.visualizer      import (
    plot_training_history, plot_predictions,
    plot_error_distribution, plot_scatter_actual_vs_pred,
    generate_report,
)
from utils.live_predictor  import LivePredictor
from models.ann_model      import StockANN


# ── Logging setup ──────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "pipeline.log"), mode="a"),
    ],
)
logger = logging.getLogger("main")


# ── Argument parser ────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Stock Market Price Prediction — ANN (Major Project)"
    )
    # Data
    p.add_argument("--ticker",   type=str,   default="AAPL",
                   help="Yahoo Finance ticker (default: AAPL)")
    p.add_argument("--period",   type=str,   default="5y",
                   help="Historical data period (default: 5y)")
    p.add_argument("--interval", type=str,   default="1d",
                   help="Data interval (default: 1d)")
    p.add_argument("--window",   type=int,   default=30,
                   help="Sliding window size in days (default: 30)")

    # Model
    p.add_argument("--hidden-layers", type=int, nargs="+", default=[128, 64, 32],
                   help="Neurons per hidden layer (default: 128 64 32)")
    p.add_argument("--lr",            type=float, default=1e-3,
                   help="Learning rate (default: 0.001)")
    p.add_argument("--epochs",        type=int,   default=150,
                   help="Max training epochs (default: 150)")
    p.add_argument("--batch-size",    type=int,   default=32,
                   help="Batch size (default: 32)")
    p.add_argument("--dropout",       type=float, default=0.2,
                   help="Dropout rate (default: 0.2)")

    # Modes
    p.add_argument("--live-only",     action="store_true",
                   help="Skip training; load saved model and run live prediction")
    p.add_argument("--live-interval", type=int, default=300,
                   help="Live prediction update interval in seconds (default: 300)")
    p.add_argument("--live-duration", type=int, default=0,
                   help="How long (s) to run live loop; 0 = indefinite (Ctrl+C to stop)")

    # Output
    p.add_argument("--output-dir",    type=str, default="reports",
                   help="Directory for charts and HTML report")
    p.add_argument("--model-dir",     type=str, default="models/saved",
                   help="Directory to save/load model artifacts")

    return p.parse_args()


# ── Pipeline functions ─────────────────────────────────────────────────────────

def run_training_pipeline(args) -> tuple:
    """Full train → evaluate → report pipeline. Returns (model, preprocessor, fetcher)."""
    ticker = args.ticker.upper()
    out_dir = os.path.join(os.path.dirname(__file__), args.output_dir, ticker)
    mdl_dir = os.path.join(os.path.dirname(__file__), args.model_dir, ticker)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(mdl_dir, exist_ok=True)

    # ── 1. Fetch data ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Fetching historical data for %s", ticker)
    fetcher = StockDataFetcher(ticker=ticker, period=args.period, interval=args.interval)
    df = fetcher.fetch()

    logger.info("Data shape: %s | Date range: %s → %s",
                df.shape, df.index[0].date(), df.index[-1].date())

    # ── 2. Preprocess ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Preprocessing — window=%d", args.window)
    preprocessor = DataPreprocessor(window_size=args.window)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare(df)

    input_dim = X_train.shape[1]
    logger.info("Input dimension: %d", input_dim)

    # ── 3. Build ANN ───────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Building ANN — layers: %s", args.hidden_layers)
    ann_config = {
        "hidden_layers": args.hidden_layers,
        "learning_rate": args.lr,
        "epochs":        args.epochs,
        "batch_size":    args.batch_size,
        "dropout_rate":  args.dropout,
    }
    model = StockANN(input_dim=input_dim, config=ann_config)

    # ── 4. Train ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Training ANN via backpropagation")
    checkpoint_path = os.path.join(mdl_dir, "best_checkpoint.keras")
    history = model.train(X_train, y_train, X_val, y_val,
                          checkpoint_path=checkpoint_path)

    # ── 5. Evaluate ────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5: Evaluating on test set")
    metrics = model.evaluate(X_test, y_test, scaler=preprocessor.target_scaler)

    print("\n" + "=" * 50)
    print(f"  EVALUATION METRICS — {ticker}")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:<35}: {v}")
    print("=" * 50 + "\n")

    # Inverse-transform for charts
    y_pred_scaled = model.predict(X_test)
    y_pred = preprocessor.target_scaler.inverse_transform(
        y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = preprocessor.target_scaler.inverse_transform(
        y_test.reshape(-1, 1)).flatten()

    # Test set dates (approximate)
    n_test = len(y_true)
    dates  = df.index[-n_test:] if len(df) >= n_test else None

    # ── 6. Visualizations ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6: Generating visualizations")
    plot_training_history(history, out_dir)
    plot_predictions(y_true, y_pred, ticker, out_dir, dates=dates)
    plot_error_distribution(y_true, y_pred, out_dir)
    plot_scatter_actual_vs_pred(y_true, y_pred, out_dir)

    # ── 7. Live snapshot prediction ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 7: Live snapshot prediction")
    live_predictor = LivePredictor(fetcher, preprocessor, model, ticker)
    live_result = live_predictor.predict_now()

    prediction_summary = {}
    if live_result:
        prediction_summary = {
            "Last Close Price":    live_result["last_close"],
            "ANN Predicted Price": live_result["predicted_price"],
            "Expected Change":     f"{live_result['change_pct']:+.3f}%",
            "Direction Signal":    live_result["direction"],
            "Prediction Time":     live_result["timestamp"],
        }
    else:
        prediction_summary = {"Status": "Live prediction unavailable (market closed / no data)"}

    # ── 8. HTML Report ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 8: Generating HTML report")
    report_path = generate_report(ticker, metrics, model.config,
                                  prediction_summary, out_dir)

    # ── 9. Save model artifacts ────────────────────────────────────────────────
    model.save(mdl_dir)
    preprocessor.save(mdl_dir)

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info("Report : %s", report_path)
    logger.info("Model  : %s", mdl_dir)

    return model, preprocessor, fetcher, live_predictor


def run_live_only(args):
    """Load a saved model and run live prediction continuously."""
    ticker  = args.ticker.upper()
    mdl_dir = os.path.join(os.path.dirname(__file__), args.model_dir, ticker)

    if not os.path.exists(mdl_dir):
        logger.error("Saved model not found at %s. Run training first.", mdl_dir)
        sys.exit(1)

    preprocessor = DataPreprocessor.load(mdl_dir)
    input_dim    = preprocessor.window_size * len(preprocessor.feature_cols)
    model        = StockANN.load(mdl_dir, input_dim=input_dim)
    fetcher      = StockDataFetcher(ticker=ticker, interval=args.interval)

    live_predictor = LivePredictor(fetcher, preprocessor, model, ticker,
                                   on_prediction=lambda r: print(
                                       f"\n[{r['timestamp']}] {r['ticker']}: "
                                       f"Last={r['last_close']} | "
                                       f"Predicted={r['predicted_price']} | "
                                       f"{r['direction']} ({r['change_pct']:+.3f}%)"
                                   ))
    live_predictor.start(interval_seconds=args.live_interval)

    try:
        duration = args.live_duration
        if duration > 0:
            logger.info("Running live predictions for %d seconds...", duration)
            time.sleep(duration)
        else:
            logger.info("Running live predictions indefinitely. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        live_predictor.stop()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Stock Market Price Prediction — ANN (Major Project)")
    logger.info("Ticker: %s | Period: %s | Window: %d",
                args.ticker, args.period, args.window)
    logger.info("=" * 60)

    if args.live_only:
        run_live_only(args)
    else:
        model, preprocessor, fetcher, live_predictor = run_training_pipeline(args)

        # Optionally start continuous live loop after training
        if args.live_duration != 0 or args.live_only:
            live_predictor.start(interval_seconds=args.live_interval)
            try:
                duration = args.live_duration
                if duration > 0:
                    time.sleep(duration)
                else:
                    while True:
                        time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Interrupted by user.")
            finally:
                live_predictor.stop()


if __name__ == "__main__":
    main()
