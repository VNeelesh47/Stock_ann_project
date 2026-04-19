"""
Visualization & Reporting Module
==================================
Generates performance charts (Matplotlib) and a full HTML report.

Charts produced:
  1. training_history.png       : Loss and MAE curves over epochs
  2. {TICKER}_predictions.png   : Actual vs Predicted price + residuals
  3. error_distribution.png     : Histogram of prediction errors
  4. scatter_actual_vs_pred.png : Scatter plot -- how close are predictions?

HTML Report includes:
  - All 4 charts
  - Evaluation metrics (MAPE, RMSE, MAE, R2, Directional Accuracy)
  - Live prediction summary
  - ANN hyperparameters
  - Architecture explanation (Why ANN? Why this design?)
  - Limitations and result interpretation

NOTE: All special characters (emojis, arrows) have been removed to ensure
compatibility with Windows cp1252 encoding. UTF-8 is explicitly set on write.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                          # non-interactive backend (works on all OS)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# -- Color Palette -------------------------------------------------------------
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
    "figure.facecolor": STYLE["bg"],
    "axes.facecolor":   STYLE["panel"],
    "axes.edgecolor":   STYLE["grid"],
    "axes.labelcolor":  STYLE["text"],
    "xtick.color":      STYLE["text"],
    "ytick.color":      STYLE["text"],
    "text.color":       STYLE["text"],
    "grid.color":       STYLE["grid"],
    "grid.linewidth":   0.5,
    "legend.facecolor": STYLE["panel"],
    "legend.edgecolor": STYLE["grid"],
    "font.family":      "monospace",
})


# -- Chart 1: Training History -------------------------------------------------

def plot_training_history(history, save_dir: str) -> str:
    """Plot Huber Loss and MAE curves for train and validation sets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ANN Training History (Backpropagation)",
                 fontsize=14, color=STYLE["text"], fontweight="bold")

    pairs  = [("loss", "Huber Loss"), ("mae", "Mean Absolute Error")]
    colors = [STYLE["accent1"], STYLE["accent3"]]

    for ax, (key, label), color in zip(axes, pairs, colors):
        ax.plot(history.history[key],          color=color,
                lw=2, label="Train")
        ax.plot(history.history[f"val_{key}"], color=STYLE["accent2"],
                lw=2, linestyle="--", label="Validation")
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


# -- Chart 2: Actual vs Predicted (Gap fixed: "No Actual vs Predicted graph") --

def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                     ticker: str, save_dir: str, dates=None) -> str:
    """
    Main result chart: actual price vs ANN predicted price on test set.
    Bottom panel shows residuals (actual - predicted).
    """
    fig = plt.figure(figsize=(16, 8))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    x = dates if dates is not None else np.arange(len(y_true))

    ax1.plot(x, y_true, color=STYLE["accent1"], lw=1.5,
             label="Actual Price", alpha=0.9)
    ax1.plot(x, y_pred, color=STYLE["accent2"], lw=1.5,
             label="ANN Predicted", alpha=0.9, linestyle="--")
    ax1.fill_between(x, y_true, y_pred,
                     alpha=0.12, color=STYLE["accent3"],
                     label="Prediction Gap")

    ax1.set_title(f"{ticker} -- Actual vs ANN Predicted Price (Test Set)",
                  fontsize=13, color=STYLE["text"], fontweight="bold")
    ax1.set_ylabel("Price")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.35)

    # Residual bars
    residuals = y_true - y_pred
    bar_colors = [STYLE["accent1"] if r >= 0 else STYLE["accent2"] for r in residuals]
    ax2.bar(x, residuals, color=bar_colors, alpha=0.7, width=0.8)
    ax2.axhline(0, color=STYLE["text"], lw=0.8, linestyle="--")
    ax2.set_ylabel("Residual")
    ax2.set_xlabel("Date" if dates is not None else "Test Sample")
    ax2.grid(True, alpha=0.35)

    plt.setp(ax1.get_xticklabels(), visible=False)

    path = os.path.join(save_dir, f"{ticker}_predictions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Predictions chart saved: %s", path)
    return path


# -- Chart 3: Error Distribution -----------------------------------------------

def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray,
                             save_dir: str) -> str:
    """
    Histogram of percentage prediction errors.
    Ideal: narrow, centered near zero.
    """
    pct_errors = ((y_pred - y_true) / (y_true + 1e-9)) * 100
    mean_err   = np.mean(pct_errors)
    std_err    = np.std(pct_errors)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(pct_errors, bins=40, color=STYLE["accent1"],
            edgecolor=STYLE["bg"], alpha=0.85, label="Error distribution")
    ax.axvline(0,        color=STYLE["accent2"], lw=1.8,
               linestyle="--", label="Zero error line")
    ax.axvline(mean_err, color=STYLE["accent3"], lw=1.8,
               linestyle="-.", label=f"Mean: {mean_err:.2f}%")

    ax.set_title(f"Prediction Error Distribution  |  Std={std_err:.2f}%",
                 fontsize=13, color=STYLE["text"])
    ax.set_xlabel("Percentage Error (%)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.35)

    path = os.path.join(save_dir, "error_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Error distribution chart saved: %s", path)
    return path


# -- Chart 4: Scatter Plot -----------------------------------------------------

def plot_scatter_actual_vs_pred(y_true: np.ndarray, y_pred: np.ndarray,
                                save_dir: str) -> str:
    """
    Scatter: actual price (X) vs predicted price (Y).
    Perfect model = all dots on the diagonal line.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.45, s=14, color=STYLE["accent1"],
               label="Predictions")
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], color=STYLE["accent2"],
            lw=1.8, linestyle="--", label="Perfect prediction line")

    ax.set_title("Actual vs Predicted -- Test Set",
                 fontsize=13, color=STYLE["text"])
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.legend()
    ax.grid(True, alpha=0.35)

    path = os.path.join(save_dir, "scatter_actual_vs_pred.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Scatter chart saved: %s", path)
    return path


# -- HTML Report (Gap fixed: "No strong insight/analysis output") --------------

def generate_report(ticker: str, metrics: dict, config: dict,
                    prediction_summary: dict, save_dir: str,
                    analysis: dict = None, history=None) -> str:
    """
    Generate a full HTML performance report with:
      - Evaluation metrics
      - Live prediction summary
      - ANN architecture explanation
      - Charts
      - Result interpretation and limitations

    All characters are ASCII/basic-latin to avoid Windows encoding errors.

    Args:
        ticker             : Stock ticker
        metrics            : Evaluation metrics dict
        config             : ANN hyperparameter config
        prediction_summary : Live prediction results
        save_dir           : Output directory
        analysis           : Optional dict with explanation notes
        history            : Keras history object (for epoch info)
    Returns:
        Path to HTML file
    """

    def _card(label, value):
        color = STYLE["accent1"]
        if "MAPE" in label and isinstance(value, (int, float)):
            if value < 5:
                color = STYLE["accent1"]
            elif value < 10:
                color = STYLE["accent3"]
            else:
                color = STYLE["accent2"]
        if "R2" in label and isinstance(value, (int, float)):
            color = STYLE["accent1"] if value > 0.9 else (
                STYLE["accent3"] if value > 0.7 else STYLE["accent2"])
        return (f'<div class="card">'
                f'<div class="card-label">{label}</div>'
                f'<div class="card-value" style="color:{color}">{value}</div>'
                f'</div>')

    def _row(k, v):
        return f"<tr><td>{k}</td><td><b>{v}</b></td></tr>"

    metric_cards = "".join([_card(k, v) for k, v in metrics.items()])
    config_rows  = "".join([_row(k, v) for k, v in config.items()])
    pred_rows    = "".join([_row(k, v) for k, v in prediction_summary.items()])

    # Analysis section
    analysis_html = ""
    if analysis:
        rows = "".join([_row(k, v) for k, v in analysis.items()])
        analysis_html = f"""
        <h2>Architecture and Analysis</h2>
        <table>
          <tr><th>Topic</th><th>Details</th></tr>
          {rows}
        </table>"""

    # Training epochs info
    epoch_info = ""
    if history:
        epochs_run = len(history.history["loss"])
        best_loss  = min(history.history["val_loss"])
        epoch_info = (f"Trained for {epochs_run} epochs | "
                      f"Best validation loss: {best_loss:.6f}")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Courier New', monospace;
    padding: 2rem;
    line-height: 1.6;
  }}
  h1 {{
    color: var(--accent1);
    font-size: 1.8rem;
    margin-bottom: 0.3rem;
    letter-spacing: 0.03em;
  }}
  h2 {{
    color: var(--accent3);
    font-size: 1.05rem;
    margin: 1.8rem 0 0.7rem;
    border-bottom: 1px solid var(--grid);
    padding-bottom: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }}
  .subtitle {{
    color: #888;
    font-size: 0.82rem;
    margin-bottom: 0.3rem;
  }}
  .epoch-info {{
    color: var(--accent3);
    font-size: 0.82rem;
    margin-bottom: 2rem;
  }}
  .metrics {{
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    margin-bottom: 1.5rem;
  }}
  .card {{
    background: var(--panel);
    border: 1px solid var(--grid);
    border-radius: 8px;
    padding: 1rem 1.5rem;
    min-width: 160px;
  }}
  .card-label {{
    font-size: 0.68rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
  }}
  .card-value {{
    font-size: 1.45rem;
    font-weight: bold;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--panel);
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 1.5rem;
  }}
  th, td {{
    padding: 0.65rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--grid);
    font-size: 0.88rem;
  }}
  th {{
    background: #12141f;
    color: var(--accent1);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }}
  td:first-child {{ color: #999; }}
  .chart-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 1rem;
  }}
  .chart-box {{
    background: var(--panel);
    border: 1px solid var(--grid);
    border-radius: 8px;
    padding: 0.5rem;
  }}
  .chart-box.wide {{ grid-column: 1 / -1; }}
  img {{ width: 100%; border-radius: 4px; }}
  .interpret {{
    background: var(--panel);
    border-left: 3px solid var(--accent3);
    padding: 1rem 1.2rem;
    border-radius: 0 8px 8px 0;
    margin-bottom: 1.5rem;
    font-size: 0.88rem;
    line-height: 1.8;
  }}
  footer {{
    margin-top: 2.5rem;
    color: #444;
    font-size: 0.72rem;
    text-align: center;
    border-top: 1px solid var(--grid);
    padding-top: 1rem;
  }}
</style>
</head>
<body>

<h1>{ticker} -- ANN Stock Price Prediction Report</h1>
<p class="subtitle">Generated: {ts} | Framework: TensorFlow/Keras | Data: yfinance</p>
<p class="epoch-info">{epoch_info}</p>

<h2>Evaluation Metrics</h2>
<div class="metrics">{metric_cards}</div>

<h2>Result Interpretation</h2>
<div class="interpret">
  <b>MAPE (Mean Absolute Percentage Error):</b> How far off predictions are on average, as a
  percentage. Under 5% is considered good for stock prediction.<br/>
  <b>RMSE (Root Mean Squared Error):</b> Average error in the same unit as price (e.g. USD).
  Larger errors are penalized more.<br/>
  <b>R2 Score:</b> Measures how well the model explains price variance.
  1.0 = perfect, 0.0 = no better than predicting the mean price.<br/>
  <b>Directional Accuracy:</b> What percentage of UP/DOWN movements the ANN called correctly.
  Above 55% is meaningful -- random guessing gives 50%.<br/>
  <b>Within 5% Accuracy:</b> Percentage of predictions that landed within 5% of the actual price.
</div>

<h2>Live Prediction Summary</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  {pred_rows}
</table>

{analysis_html}

<h2>ANN Hyperparameters</h2>
<table>
  <tr><th>Parameter</th><th>Value</th></tr>
  {config_rows}
</table>

<h2>Limitations</h2>
<div class="interpret">
  1. ANN cannot predict sudden price shocks caused by news, earnings surprises,
     or geopolitical events -- these are not present in historical price data.<br/>
  2. The model learns from past patterns. When market conditions change
     significantly (e.g. from bull to bear market), the model should be retrained.<br/>
  3. Predictions are for the next trading day. Multi-day or multi-week forecasts
     will have increasing error.<br/>
  4. Volume spikes or circuit breakers can break the model's assumptions.
</div>

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

<footer>
  Stock Market Price Prediction Using Artificial Neural Network (ANN) on Live Data from yfinance
  | Major Project | Python + TensorFlow/Keras + yfinance + scikit-learn + Matplotlib
</footer>
</body>
</html>"""

    report_path = os.path.join(save_dir, f"{ticker}_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("HTML report saved: %s", report_path)
    return report_path