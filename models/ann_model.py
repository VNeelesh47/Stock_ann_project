"""
ANN Model -- Stock Market Price Prediction
==========================================
Architecture : Multi-layer Feedforward Neural Network
Training     : Backpropagation (Adam optimizer, Huber Loss)
Framework    : TensorFlow / Keras

WHY ANN?
--------
Stock prices follow non-linear, complex patterns that simple models
(like linear regression or moving averages) cannot capture.
ANNs learn these hidden patterns by adjusting weights through
backpropagation across thousands of training examples.

WHY THIS ARCHITECTURE?
----------------------
Input Layer
  -> Receives a flat vector: (window_size x n_features)
     e.g. 30 days x 19 features = 570 inputs

Hidden Layer 1 (128 neurons)
  -> Learns broad low-level patterns (price momentum, volume trends)
  -> BatchNorm: stabilizes and speeds up training
  -> Dropout(20%): randomly disables neurons to prevent memorization

Hidden Layer 2 (64 neurons)
  -> Learns mid-level combinations (RSI + MACD interaction etc.)

Hidden Layer 3 (32 neurons)
  -> Compresses into high-level signal (buy/sell pressure summary)

Output Layer (1 neuron, linear activation)
  -> Outputs a single predicted price value

LOSS FUNCTION: Huber Loss
  -> Better than MSE for stocks because stock prices have sudden spikes.
     MSE heavily penalizes those spikes, destabilizing training.
     Huber Loss is like MSE for small errors, MAE for large errors.

OPTIMIZER: Adam
  -> Adaptive learning rate. Works better than plain SGD for financial data
     which has irregular, non-stationary patterns.

EARLY STOPPING + ReduceLROnPlateau
  -> Stops training when validation loss stops improving (avoids overfitting)
  -> Reduces learning rate when stuck (helps escape local minima)

WHAT CAN FAIL:
  -> ANN cannot predict sudden news events (earnings surprise, geopolitical shock)
  -> Requires retraining when market regime changes (e.g. bull to bear market)
  -> Predictions become less accurate during high volatility periods
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import logging
import os
import json

logger = logging.getLogger(__name__)


class StockANN:
    """
    Multi-layer Feedforward ANN trained via Backpropagation.

    Architecture:
        Input -> Dense(128)+BN+Drop -> Dense(64)+BN+Drop -> Dense(32)+BN+Drop -> Output(1)
    """

    def __init__(self, input_dim: int, config: dict = None):
        self.input_dim = input_dim
        self.config = {
            "hidden_layers":      [128, 64, 32],
            "activation":         "relu",
            "output_activation":  "linear",
            "dropout_rate":       0.2,
            "l2_lambda":          1e-4,
            "learning_rate":      1e-3,
            "batch_size":         32,
            "epochs":             150,
            "patience":           20,
            "reduce_lr_patience": 10,
            "reduce_lr_factor":   0.5,
            "min_lr":             1e-6,
        }
        if config:
            self.config.update(config)
        self.model   = None
        self.history = None
        self._build()

    # -- Architecture ----------------------------------------------------------
    def _build(self):
        """Build feedforward ANN. Each hidden layer uses ReLU + BatchNorm + Dropout."""
        tf.random.set_seed(42)

        inputs = keras.Input(shape=(self.input_dim,), name="price_features")
        x = inputs

        for i, units in enumerate(self.config["hidden_layers"]):
            x = layers.Dense(
                units,
                activation=self.config["activation"],
                kernel_regularizer=regularizers.l2(self.config["l2_lambda"]),
                kernel_initializer="he_normal",
                name=f"hidden_{i+1}",
            )(x)
            x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
            x = layers.Dropout(self.config["dropout_rate"], name=f"dropout_{i+1}")(x)

        output = layers.Dense(
            1,
            activation=self.config["output_activation"],
            name="price_output",
        )(x)

        self.model = keras.Model(inputs=inputs, outputs=output, name="StockANN")
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config["learning_rate"]),
            loss="huber",
            metrics=["mae"],
        )

        # Print architecture summary
        layer_info = []
        for i, units in enumerate(self.config["hidden_layers"]):
            layer_info.append(f"Hidden {i+1}: {units} neurons (ReLU + BatchNorm + Dropout)")
        layer_info.append("Output: 1 neuron (Linear)")

        logger.info("ANN Architecture:")
        for info in layer_info:
            logger.info("  -> %s", info)
        logger.info("Loss: Huber | Optimizer: Adam | Input dim: %d", self.input_dim)

    # -- Training (Backpropagation) --------------------------------------------
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              checkpoint_path: str = None):
        """
        Train via backpropagation.
        Backpropagation:
          1. Forward pass: input -> layers -> prediction
          2. Compute loss (Huber) between prediction and actual price
          3. Backward pass: compute gradient of loss w.r.t. every weight
          4. Adam optimizer updates weights in the direction that reduces loss
          5. Repeat for all batches, all epochs
        """
        cbs = [
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config["patience"],
                restore_best_weights=True,
                verbose=1,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=self.config["reduce_lr_factor"],
                patience=self.config["reduce_lr_patience"],
                min_lr=self.config["min_lr"],
                verbose=1,
            ),
        ]
        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            cbs.append(
                callbacks.ModelCheckpoint(
                    checkpoint_path,
                    monitor="val_loss",
                    save_best_only=True,
                    verbose=1,
                )
            )

        logger.info("Training ANN via backpropagation...")
        logger.info("Train samples: %d | Val samples: %d", len(X_train), len(X_val))

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            callbacks=cbs,
            verbose=1,
        )

        epochs_run = len(self.history.history["loss"])
        best_val   = min(self.history.history["val_loss"])
        logger.info("Training complete. Epochs run: %d | Best val loss: %.6f",
                    epochs_run, best_val)
        return self.history

    # -- Inference -------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X, verbose=0).flatten()

    # -- Evaluation (Gap fixed: "Proper evaluation metrics missing or weak") ---
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 scaler=None) -> dict:
        """
        Compute full set of evaluation metrics on test data.

        Metrics returned:
          MAPE (%)             : Mean Absolute Percentage Error
                                 Lower is better. < 5% is good for stocks.
          RMSE                 : Root Mean Squared Error (same units as price)
          MAE                  : Mean Absolute Error (average price deviation)
          R2 Score             : 1.0 = perfect, 0 = no better than mean
          Directional Accuracy : % of times model correctly called UP or DOWN
          Max Error            : Worst single prediction error
          Within 5% Accuracy   : % of predictions within 5% of actual price
        """
        y_pred_scaled = self.predict(X_test)

        if scaler is not None:
            y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        else:
            y_pred = y_pred_scaled
            y_true = y_test

        mape  = mean_absolute_percentage_error(y_true, y_pred) * 100
        rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
        mae   = np.mean(np.abs(y_true - y_pred))
        r2    = r2_score(y_true, y_pred)

        # Directional accuracy
        if len(y_true) > 1:
            actual_dir = np.sign(np.diff(y_true))
            pred_dir   = np.sign(np.diff(y_pred))
            dir_acc    = np.mean(actual_dir == pred_dir) * 100
        else:
            dir_acc = float("nan")

        # Max single error
        max_err = np.max(np.abs(y_true - y_pred))

        # % of predictions within 5% of actual
        pct_errors  = np.abs((y_pred - y_true) / (y_true + 1e-9)) * 100
        within_5pct = np.mean(pct_errors <= 5.0) * 100

        metrics = {
            "MAPE (%)":                  round(mape, 4),
            "RMSE":                      round(rmse, 4),
            "MAE":                       round(mae, 4),
            "R2 Score":                  round(r2, 4),
            "Directional Accuracy (%)":  round(dir_acc, 2),
            "Max Single Error":          round(max_err, 4),
            "Within 5% Accuracy (%)":    round(within_5pct, 2),
        }

        logger.info("Evaluation complete: %s", metrics)
        return metrics

    # -- Persistence -----------------------------------------------------------
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "ann_model.keras"))
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str, input_dim: int):
        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            config = json.load(f)
        obj = cls(input_dim=input_dim, config=config)
        obj.model = keras.models.load_model(os.path.join(path, "ann_model.keras"))
        logger.info("Model loaded from %s", path)
        return obj