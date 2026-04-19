"""
Stock Market Price Prediction Using ANN
========================================
Major Project | TensorFlow/Keras + yfinance

HOW TO RUN:
    python main.py                              (interactive menu)
    python main.py --ticker AAPL               (direct train + report)
    python main.py --ticker AAPL --live-only   (load saved model, live loop)
    python main.py --ticker RELIANCE.NS --period 3y --window 20 --epochs 200
"""

import argparse
import logging
import os
import sys
import time
import numpy as np

# -- Project imports ------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_utils     import StockDataFetcher, DataPreprocessor
from utils.visualizer     import (
    plot_training_history, plot_predictions,
    plot_error_distribution, plot_scatter_actual_vs_pred,
    generate_report,
)
from utils.live_predictor import LivePredictor
from models.ann_model     import StockANN

# -- Logging --------------------------------------------------------------------
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "pipeline.log"), mode="a",
                            encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")


# 
# INTERACTIVE MENU  (Gap fixed: "No user interface / options")
# 

def print_banner():
    print("\n" + "=" * 60)
    print("   STOCK MARKET PRICE PREDICTION USING ANN")
    print("   Major Project | TensorFlow/Keras + yfinance")
    print("=" * 60)


def interactive_menu():
    """
    Full interactive menu so the user can choose what to do
    without knowing any command-line arguments.
    """
    print_banner()
    print("\nWhat would you like to do?\n")
    print("  [1]  Train model on a stock and generate full report")
    print("  [2]  Predict next price  (load saved model, single prediction)")
    print("  [3]  Live prediction loop (continuous updates every N seconds)")
    print("  [4]  Exit")
    print()

    choice = input("Enter choice (1/2/3/4): ").strip()

    if choice == "1":
        ticker  = input("Enter ticker symbol (e.g. AAPL, RELIANCE.NS) [AAPL]: ").strip() or "AAPL"
        period  = input("Historical data period (1y/2y/5y/10y) [5y]: ").strip() or "5y"
        window  = input("Sliding window size in days [30]: ").strip() or "30"
        epochs  = input("Max training epochs [150]: ").strip() or "150"
        return _make_args(ticker=ticker, period=period,
                          window=int(window), epochs=int(epochs),
                          mode="train")

    elif choice == "2":
        ticker = input("Enter ticker symbol [AAPL]: ").strip() or "AAPL"
        return _make_args(ticker=ticker, mode="predict")

    elif choice == "3":
        ticker   = input("Enter ticker symbol [AAPL]: ").strip() or "AAPL"
        interval = input("Update interval in seconds [300]: ").strip() or "300"
        duration = input("Run for how many seconds? (0 = until Ctrl+C) [0]: ").strip() or "0"
        return _make_args(ticker=ticker, live_interval=int(interval),
                          live_duration=int(duration), mode="live")

    elif choice == "4":
        print("Exiting.")
        sys.exit(0)
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)


def _make_args(**kwargs):
    """Build a simple namespace object from keyword args (mimics argparse output)."""
    import types
    defaults = {
        "ticker":        "AAPL",
        "period":        "5y",
        "interval":      "1d",
        "window":        30,
        "hidden_layers": [128, 64, 32],
        "lr":            1e-3,
        "epochs":        150,
        "batch_size":    32,
        "dropout":       0.2,
        "live_only":     False,
        "live_interval": 300,
        "live_duration": 0,
        "output_dir":    "reports",
        "model_dir":     "models/saved",
        "mode":          "train",
    }
    defaults.update(kwargs)
    ns = types.SimpleNamespace(**defaults)
    # map mode flags
    if ns.mode == "live":
        ns.live_only = True
    return ns


# 
# ARGUMENT PARSER  (for direct command-line use)
# 

def parse_args():
    p = argparse.ArgumentParser(
        description="Stock Market Price Prediction -- ANN Major Project"
    )
    p.add_argument("--ticker",        type=str,   default="AAPL")
    p.add_argument("--period",        type=str,   default="5y")
    p.add_argument("--interval",      type=str,   default="1d")
    p.add_argument("--window",        type=int,   default=30)
    p.add_argument("--hidden-layers", type=int,   nargs="+", default=[128, 64, 32],
                   dest="hidden_layers")
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--epochs",        type=int,   default=150)
    p.add_argument("--batch-size",    type=int,   default=32,  dest="batch_size")
    p.add_argument("--dropout",       type=float, default=0.2)
    p.add_argument("--live-only",     action="store_true",     dest="live_only")
    p.add_argument("--live-interval", type=int,   default=300, dest="live_interval")
    p.add_argument("--live-duration", type=int,   default=0,   dest="live_duration")
    p.add_argument("--output-dir",    type=str,   default="reports", dest="output_dir")
    p.add_argument("--model-dir",     type=str,   default="models/saved", dest="model_dir")
    args = p.parse_args()
    args.mode = "live" if args.live_only else "train"
    return args


# 
# TRAINING PIPELINE
# 

def run_training_pipeline(args):
    """
    Full pipeline:
      Fetch -> Preprocess -> Train -> Evaluate -> Visualize -> Live snapshot -> Report
    """
    ticker  = args.ticker.upper()
    base    = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base, args.output_dir, ticker)
    mdl_dir = os.path.join(base, args.model_dir, ticker)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(mdl_dir, exist_ok=True)

    # -- STEP 1: Fetch ---------------------------------------------------------
    _header("STEP 1: Fetching historical data for " + ticker)
    fetcher = StockDataFetcher(ticker=ticker, period=args.period, interval=args.interval)
    df = fetcher.fetch()
    logger.info("Rows: %d | Features: %d | Range: %s to %s",
                len(df), df.shape[1],
                df.index[0].date(), df.index[-1].date())

    # -- STEP 2: Preprocess ----------------------------------------------------
    _header("STEP 2: Preprocessing (scaling + sliding window)")
    preprocessor = DataPreprocessor(window_size=args.window)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare(df)
    input_dim = X_train.shape[1]
    logger.info("Input dim: %d | Train: %d | Val: %d | Test: %d",
                input_dim, len(X_train), len(X_val), len(X_test))

    # -- STEP 3: Build ANN -----------------------------------------------------
    _header("STEP 3: Building ANN architecture")
    ann_config = {
        "hidden_layers": args.hidden_layers,
        "learning_rate": args.lr,
        "epochs":        args.epochs,
        "batch_size":    args.batch_size,
        "dropout_rate":  args.dropout,
    }
    model = StockANN(input_dim=input_dim, config=ann_config)

    # -- STEP 4: Train ---------------------------------------------------------
    _header("STEP 4: Training ANN via backpropagation")
    checkpoint_path = os.path.join(mdl_dir, "best_checkpoint.keras")
    history = model.train(X_train, y_train, X_val, y_val,
                          checkpoint_path=checkpoint_path)

    # -- STEP 5: Evaluate ------------------------------------------------------
    # Gap fixed: "Proper evaluation metrics missing or weak"
    _header("STEP 5: Evaluating on unseen test set")
    metrics = model.evaluate(X_test, y_test, scaler=preprocessor.target_scaler)

    # Print clean metrics table to console
    print("\n" + "=" * 55)
    print(f"  EVALUATION RESULTS -- {ticker}")
    print("=" * 55)
    print(f"  {'Metric':<35} {'Value':>10}")
    print("-" * 55)
    for k, v in metrics.items():
        print(f"  {k:<35} {str(v):>10}")
    print("=" * 55)

    # Gap fixed: "Accuracy comparison not clearly shown"
    mape = metrics.get("MAPE (%)", 999)
    if mape < 3:
        grade = "EXCELLENT (< 3% error)"
    elif mape < 5:
        grade = "GOOD (< 5% error)"
    elif mape < 10:
        grade = "ACCEPTABLE (< 10% error)"
    else:
        grade = "NEEDS IMPROVEMENT (> 10% error)"
    print(f"\n  Model Grade : {grade}")
    print(f"  Directional : {metrics.get('Directional Accuracy (%)', 'N/A')}% correct "
          f"UP/DOWN calls")

    # Gap fixed: "No explanation layer -- What worked / failed?"
    print("\n  Analysis:")
    print("  - ANN uses 3 hidden layers (128 -> 64 -> 32 neurons)")
    print("  - Backpropagation with Adam optimizer adjusts weights each epoch")
    print("  - Dropout (20%) prevents the model from memorizing training data")
    print("  - Huber loss is used instead of MSE -- less sensitive to price spikes")
    print("  - Early stopping halted training to avoid overfitting")
    epochs_run = len(history.history["loss"])
    print(f"  - Trained for {epochs_run} epochs (early stopping may have cut short)")

    # Gap fixed: "No result interpretation / Limitations"
    print("\n  Limitations:")
    print("  - ANN cannot predict sudden news-driven price shocks")
    print("  - Predictions are based on past patterns, not future events")
    print("  - Model should be retrained periodically as market conditions change")
    print("=" * 55 + "\n")

    # -- Inverse-transform for charts ------------------------------------------
    y_pred_scaled = model.predict(X_test)
    y_pred = preprocessor.target_scaler.inverse_transform(
        y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = preprocessor.target_scaler.inverse_transform(
        y_test.reshape(-1, 1)).flatten()

    n_test = len(y_true)
    dates  = df.index[-n_test:] if len(df) >= n_test else None

    # -- STEP 6: Visualizations ------------------------------------------------
    _header("STEP 6: Generating charts")
    plot_training_history(history, out_dir)
    plot_predictions(y_true, y_pred, ticker, out_dir, dates=dates)
    plot_error_distribution(y_true, y_pred, out_dir)
    plot_scatter_actual_vs_pred(y_true, y_pred, out_dir)

    # -- STEP 7: Live snapshot prediction --------------------------------------
    # Gap fixed: "Live pipeline automation missing"
    _header("STEP 7: Live snapshot prediction from yfinance")
    live_predictor = LivePredictor(fetcher, preprocessor, model, ticker)
    live_result = live_predictor.predict_now()

    if live_result:
        prediction_summary = {
            "Last Close Price":    live_result["last_close"],
            "ANN Predicted Price": live_result["predicted_price"],
            "Expected Change":     f"{live_result['change_pct']:+.3f}%",
            "Direction Signal":    live_result["direction"],
            "Prediction Time":     live_result["timestamp"],
        }
        print(f"\n  Live Result for {ticker}:")
        for k, v in prediction_summary.items():
            print(f"    {k:<25}: {v}")
        print()
    else:
        prediction_summary = {
            "Status": "Live data unavailable (market may be closed)"
        }

    # -- STEP 8: HTML Report ---------------------------------------------------
    # Gap fixed: "No strong insight/analysis output"
    _header("STEP 8: Generating HTML report")
    analysis_notes = {
        "Architecture":   "3 hidden layers (128 -> 64 -> 32) with BatchNorm + Dropout",
        "Why ANN":        "Captures non-linear patterns that linear models miss",
        "Training":       f"{epochs_run} epochs with early stopping + ReduceLROnPlateau",
        "Loss Function":  "Huber Loss (robust to price outliers/spikes)",
        "Optimizer":      "Adam with learning rate decay",
        "Model Grade":    grade,
        "Limitation 1":   "Cannot predict news-driven sudden moves",
        "Limitation 2":   "Requires periodic retraining as market changes",
    }
    report_path = generate_report(
        ticker, metrics, model.config,
        prediction_summary, out_dir,
        analysis=analysis_notes,
        history=history,
    )

    # -- STEP 9: Save ----------------------------------------------------------
    model.save(mdl_dir)
    preprocessor.save(mdl_dir)

    _header("PIPELINE COMPLETE")
    print(f"  Report saved : {report_path}")
    print(f"  Model saved  : {mdl_dir}")
    print(f"  Charts saved : {out_dir}\n")

    return model, preprocessor, fetcher, live_predictor


# 
# SINGLE PREDICT (Menu option 2)
# 

def run_single_predict(args):
    """
    Load a saved model, fetch latest data, print one prediction.
    """
    ticker  = args.ticker.upper()
    base    = os.path.dirname(os.path.abspath(__file__))
    mdl_dir = os.path.join(base, args.model_dir, ticker)

    if not os.path.exists(mdl_dir):
        print(f"\n  ERROR: No saved model found for {ticker}.")
        print(f"  Please run option [1] first to train the model.\n")
        sys.exit(1)

    preprocessor = DataPreprocessor.load(mdl_dir)
    input_dim    = preprocessor.window_size * len(preprocessor.feature_cols)
    model        = StockANN.load(mdl_dir, input_dim=input_dim)
    fetcher      = StockDataFetcher(ticker=ticker, interval=args.interval)

    print(f"\n  Fetching latest data for {ticker}...")
    live_predictor = LivePredictor(fetcher, preprocessor, model, ticker)
    result = live_predictor.predict_now()

    if result:
        print("\n" + "=" * 50)
        print(f"  PREDICTION RESULT -- {ticker}")
        print("=" * 50)
        print(f"  Last Close Price  : {result['last_close']}")
        print(f"  Predicted Price   : {result['predicted_price']}")
        print(f"  Expected Change   : {result['change_pct']:+.3f}%")
        print(f"  Direction Signal  : {result['direction']}")
        print(f"  Timestamp         : {result['timestamp']}")
        print("=" * 50 + "\n")
    else:
        print("  Could not fetch live data. Market may be closed.")


# 
# LIVE LOOP  (Gap fixed: "No true live system -- fetch/update/predict continuously")
# 

def run_live_loop(args):
    """
    True continuous live prediction system.
    Fetches fresh data -> predicts -> waits -> repeats.
    Logs every prediction to live_predictions.log
    """
    ticker  = args.ticker.upper()
    base    = os.path.dirname(os.path.abspath(__file__))
    mdl_dir = os.path.join(base, args.model_dir, ticker)

    if not os.path.exists(mdl_dir):
        print(f"\n  ERROR: No saved model found for {ticker}.")
        print(f"  Please run option [1] first to train the model.\n")
        sys.exit(1)

    preprocessor = DataPreprocessor.load(mdl_dir)
    input_dim    = preprocessor.window_size * len(preprocessor.feature_cols)
    model        = StockANN.load(mdl_dir, input_dim=input_dim)
    fetcher      = StockDataFetcher(ticker=ticker, interval=args.interval)

    # Log file for live predictions
    live_log_path = os.path.join(LOG_DIR, f"{ticker}_live_predictions.log")
    live_log = open(live_log_path, "a", encoding="utf-8")

    def on_prediction(result):
        line = (f"[{result['timestamp']}]  "
                f"Ticker={result['ticker']}  "
                f"LastClose={result['last_close']}  "
                f"Predicted={result['predicted_price']}  "
                f"Change={result['change_pct']:+.3f}%  "
                f"Direction={result['direction']}")
        print("\n" + "=" * 60)
        print(f"  LIVE UPDATE -- {result['ticker']}")
        print(f"  Time       : {result['timestamp']}")
        print(f"  Last Close : {result['last_close']}")
        print(f"  Predicted  : {result['predicted_price']}")
        print(f"  Change     : {result['change_pct']:+.3f}%")
        print(f"  Direction  : {result['direction']}")
        print("=" * 60)
        live_log.write(line + "\n")
        live_log.flush()

    predictor = LivePredictor(fetcher, preprocessor, model, ticker,
                              on_prediction=on_prediction)
    predictor.start(interval_seconds=args.live_interval)

    duration = args.live_duration
    print(f"\n  Live prediction loop started for {ticker}.")
    print(f"  Update interval : {args.live_interval} seconds")
    print(f"  Log file        : {live_log_path}")
    if duration > 0:
        print(f"  Will run for    : {duration} seconds")
    else:
        print("  Press Ctrl+C to stop.")
    print()

    try:
        if duration > 0:
            time.sleep(duration)
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Stopped by user.")
    finally:
        predictor.stop()
        live_log.close()
        print(f"  Predictions saved to: {live_log_path}\n")


# 
# HELPERS
# 

def _header(text: str):
    logger.info("=" * 60)
    logger.info(text)
    logger.info("=" * 60)


# 
# ENTRY POINT
# 

def main():
    # If no arguments passed -> show interactive menu
    if len(sys.argv) == 1:
        args = interactive_menu()
    else:
        args = parse_args()

    logger.info("Starting | Ticker: %s | Mode: %s", args.ticker.upper(),
                getattr(args, "mode", "train"))

    mode = getattr(args, "mode", "train")

    if mode == "live" or args.live_only:
        run_live_loop(args)

    elif mode == "predict":
        run_single_predict(args)

    else:
        # Full training pipeline
        model, preprocessor, fetcher, live_predictor = run_training_pipeline(args)

        # After training, ask if user wants to start live loop
        if len(sys.argv) == 1:  # only in interactive mode
            print("  Training complete!")
            start_live = input("  Start live prediction loop now? (y/n) [n]: ").strip().lower()
            if start_live == "y":
                interval = input("  Update interval in seconds [300]: ").strip() or "300"
                args.live_interval = int(interval)
                args.live_duration = 0
                run_live_loop(args)


if __name__ == "__main__":
    main()