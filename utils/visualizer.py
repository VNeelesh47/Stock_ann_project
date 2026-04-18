"""
Visualization & Reporting Module
Generates performance charts and saves HTML/PNG reports
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                          # non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

STYLE = {
    "bg":      "#0f1117",
    "panel":   "#1a1d27",
    "accent1": "#00d4aa",
    "accent2": "#ff6b6b",
    "accent3": "#ffd166",
    "text":    "#e0e0e0",
    "grid":    "#2a2d3a",
}

plt.rcParams.update({
    "figure.facecolor":  STYLE["bg"],
    "axes.facecolor":    STYLE["panel"],
    "axes.edgecolor":    STYLE["grid"],
    "axes.labelcolor":   STYLE["text"],
    "xtick.color":       STYLE["text"],
    "ytick.color":       STYLE["text"],
    "text.color":        STYLE["text"],
    "grid.color":        STYLE["grid"],
    "grid.linewidth":    0.5,
    "legend.facecolor":  STYLE["panel"],
    "legend.edgecolor":  STYLE["grid"],
    "font.family":       "monospace",
})


def plot_training_history(history, save_dir: str):
    """Plot loss and MAE curves from Keras training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ANN Training History", fontsize=14, color=STYLE["text"], fontweight="bold")

    metrics = [("loss", "Huber Loss"), ("mae", "Mean Absolute Error")]
    colors  = [STYLE["accent1"], STYLE["accent3"]]

    for ax, (key, label), color in zip(axes, metrics, colors):
        ax.plot(history.history[key],     color=color,          lw=2, label="Train")
        ax.plot(history.history[f"val_{key}"], color=STYLE["accent2"], lw=2,
                linestyle="--", label="Validation")
        ax.set_title(label, color=STYLE["text"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.4)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    path = os.path.join(save_dir, "training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Training history chart saved: %s", path)
    return path


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                     ticker: str, save_dir: str, dates=None):
    """Plot actual vs predicted prices on test data."""
    fig = plt.figure(figsize=(16, 8))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.1)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    x = dates if dates is not None else np.arange(len(y_true))

    ax1.plot(x, y_true, color=STYLE["accent1"], lw=1.5, label="Actual Price", alpha=0.9)
    ax1.plot(x, y_pred, color=STYLE["accent2"], lw=1.5, label="ANN Predicted", alpha=0.9,
             linestyle="--")
    ax1.fill_between(x, y_true, y_pred, alpha=0.15, color=STYLE["accent3"])
    ax1.set_title(f"{ticker} — Actual vs ANN Predicted Stock Price",
                  fontsize=13, color=STYLE["text"], fontweight="bold")
    ax1.set_ylabel("Price (USD / Local Currency)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.35)

    # Residuals
    residuals = y_true - y_pred
    ax2.bar(x, residuals, color=np.where(residuals >= 0, STYLE["accent1"], STYLE["accent2"]),
            alpha=0.7, width=0.8)
    ax2.axhline(0, color=STYLE["text"], lw=0.8, linestyle="--")
    ax2.set_ylabel("Residual")
    ax2.set_xlabel("Date" if dates is not None else "Test Sample Index")
    ax2.grid(True, alpha=0.35)

    plt.setp(ax1.get_xticklabels(), visible=False)

    path = os.path.join(save_dir, f"{ticker}_predictions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Prediction chart saved: %s", path)
    return path


def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray, save_dir: str):
    """Histogram of percentage errors."""
    pct_errors = ((y_pred - y_true) / (y_true + 1e-9)) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(pct_errors, bins=40, color=STYLE["accent1"], edgecolor=STYLE["bg"], alpha=0.85)
    ax.axvline(0, color=STYLE["accent2"], lw=1.5, linestyle="--", label="Zero Error")
    ax.axvline(np.mean(pct_errors), color=STYLE["accent3"], lw=1.5,
               linestyle="-.", label=f"Mean Error: {np.mean(pct_errors):.2f}%")
    ax.set_title("Prediction Error Distribution", fontsize=13, color=STYLE["text"])
    ax.set_xlabel("Percentage Error (%)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.35)

    path = os.path.join(save_dir, "error_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Error distribution chart saved: %s", path)
    return path


def plot_scatter_actual_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, save_dir: str):
    """Scatter plot of actual vs predicted prices."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.45, s=15, color=STYLE["accent1"])
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], color=STYLE["accent2"], lw=1.5, linestyle="--", label="Perfect Fit")
    ax.set_title("Actual vs Predicted (Test Set)", fontsize=13, color=STYLE["text"])
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.legend()
    ax.grid(True, alpha=0.35)

    path = os.path.join(save_dir, "scatter_actual_vs_pred.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Scatter chart saved: %s", path)
    return path


def generate_report(ticker: str, metrics: dict, config: dict,
                    prediction_summary: dict, save_dir: str) -> str:
    """
    Generate a self-contained HTML performance report.

    Args:
        ticker             : Stock ticker symbol
        metrics            : Evaluation metrics dict
        config             : ANN hyperparameter config
        prediction_summary : {'last_actual', 'last_predicted', 'next_predicted'}
        save_dir           : Output directory
    Returns:
        Path to the HTML file
    """
    def _metric_card(label, value, unit=""):
        color = STYLE["accent1"]
        if "MAPE" in label and isinstance(value, (int, float)):
            color = STYLE["accent1"] if value < 5 else (STYLE["accent3"] if value < 10 else STYLE["accent2"])
        return f"""
        <div class="card">
            <div class="card-label">{label}</div>
            <div class="card-value" style="color:{color}">{value}{unit}</div>
        </div>"""

    def _row(k, v):
        return f"<tr><td>{k}</td><td><b>{v}</b></td></tr>"

    metric_cards = "".join([_metric_card(k, v) for k, v in metrics.items()])
    config_rows  = "".join([_row(k, v) for k, v in config.items()])
    pred_rows    = "".join([_row(k, v) for k, v in prediction_summary.items()])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{ticker} ANN Stock Prediction Report</title>
<style>
  :root {{
    --bg: {STYLE['bg']};
    --panel: {STYLE['panel']};
    --accent1: {STYLE['accent1']};
    --accent2: {STYLE['accent2']};
    --accent3: {STYLE['accent3']};
    --text: {STYLE['text']};
    --grid: {STYLE['grid']};
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Courier New', monospace;
         padding: 2rem; }}
  h1 {{ color: var(--accent1); font-size: 1.8rem; margin-bottom: 0.3rem; }}
  h2 {{ color: var(--accent3); font-size: 1.1rem; margin: 1.5rem 0 0.7rem; border-bottom:
        1px solid var(--grid); padding-bottom: 0.3rem; }}
  .subtitle {{ color: #888; font-size: 0.85rem; margin-bottom: 2rem; }}
  .metrics {{ display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1.5rem; }}
  .card {{ background: var(--panel); border: 1px solid var(--grid); border-radius: 8px;
           padding: 1rem 1.5rem; min-width: 160px; }}
  .card-label {{ font-size: 0.72rem; color: #888; text-transform: uppercase;
                 letter-spacing: 0.1em; margin-bottom: 0.4rem; }}
  .card-value {{ font-size: 1.5rem; font-weight: bold; }}
  table {{ width: 100%; border-collapse: collapse; background: var(--panel);
           border-radius: 8px; overflow: hidden; margin-bottom: 1.5rem; }}
  th, td {{ padding: 0.7rem 1rem; text-align: left; border-bottom: 1px solid var(--grid); }}
  th {{ background: #12141f; color: var(--accent1); font-size: 0.8rem;
        text-transform: uppercase; letter-spacing: 0.08em; }}
  td:first-child {{ color: #888; }}
  .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem; }}
  .chart-box {{ background: var(--panel); border: 1px solid var(--grid); border-radius: 8px;
                padding: 0.5rem; }}
  .chart-box.wide {{ grid-column: 1 / -1; }}
  img {{ width: 100%; border-radius: 4px; }}
  footer {{ margin-top: 2rem; color: #555; font-size: 0.75rem; text-align: center; }}
</style>
</head>
<body>
<h1>📈 {ticker} — ANN Stock Price Prediction</h1>
<p class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
   Model: Multi-layer Feedforward ANN (Backpropagation) | Data Source: yfinance</p>

<h2>Performance Metrics</h2>
<div class="metrics">{metric_cards}</div>

<h2>Live Prediction Summary</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  {pred_rows}
</table>

<h2>ANN Hyperparameters</h2>
<table>
  <tr><th>Parameter</th><th>Value</th></tr>
  {config_rows}
</table>

<h2>Visualizations</h2>
<div class="chart-grid">
  <div class="chart-box wide">
    <img src="{ticker}_predictions.png" alt="Actual vs Predicted"/>
  </div>
  <div class="chart-box">
    <img src="training_history.png" alt="Training History"/>
  </div>
  <div class="chart-box">
    <img src="error_distribution.png" alt="Error Distribution"/>
  </div>
  <div class="chart-box">
    <img src="scatter_actual_vs_pred.png" alt="Scatter Plot"/>
  </div>
</div>

<footer>Stock Market Price Prediction Using ANN | Major Project | yfinance + TensorFlow/Keras</footer>
</body>
</html>"""

    report_path = os.path.join(save_dir, f"{ticker}_report.html")
    with open(report_path, "w") as f:
        f.write(html)
    logger.info("HTML report saved: %s", report_path)
    return report_path
