"""
ANN Model for Stock Price Prediction
Multi-layer Feedforward Neural Network with Backpropagation
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import logging
import os
import json

logger = logging.getLogger(__name__)


class StockANN:
    """
    Multi-layer Feedforward ANN model using Keras/TensorFlow.
    Architecture: Input → Hidden Layers (with Dropout + BatchNorm) → Output
    Training: Backpropagation with Adam optimizer
    """

    def __init__(self, input_dim: int, config: dict = None):
        """
        Args:
            input_dim   : Number of input features (sliding window size × feature count)
            config      : Hyperparameter dictionary (overrides defaults)
        """
        self.input_dim = input_dim
        self.config = {
            "hidden_layers": [128, 64, 32],   # neurons per hidden layer
            "activation": "relu",
            "output_activation": "linear",
            "dropout_rate": 0.2,
            "l2_lambda": 1e-4,
            "learning_rate": 1e-3,
            "batch_size": 32,
            "epochs": 150,
            "patience": 20,                   # early stopping patience
            "reduce_lr_patience": 10,
            "reduce_lr_factor": 0.5,
            "min_lr": 1e-6,
        }
        if config:
            self.config.update(config)

        self.model = None
        self.history = None
        self._build()

    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------
    def _build(self):
        """Build feedforward ANN with backpropagation-ready architecture."""
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

        output = layers.Dense(1, activation=self.config["output_activation"], name="price_output")(x)

        self.model = keras.Model(inputs=inputs, outputs=output, name="StockANN")
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config["learning_rate"]),
            loss="huber",          # robust to outliers (better than MSE for finance)
            metrics=["mae"],
        )
        logger.info("ANN architecture built successfully.")
        logger.info(self.model.summary())

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray, checkpoint_path: str = None):
        """
        Train the ANN using backpropagation.

        Args:
            X_train, y_train : Training data
            X_val, y_val     : Validation data
            checkpoint_path  : Path to save best model weights
        Returns:
            history object
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

        logger.info("Starting ANN training with backpropagation...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            callbacks=cbs,
            verbose=1,
        )
        logger.info("Training complete.")
        return self.history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return raw (scaled) predictions."""
        return self.model.predict(X, verbose=0).flatten()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 scaler=None) -> dict:
        """
        Compute MAPE, RMSE, and directional accuracy on test data.

        Args:
            X_test, y_test : Test arrays (scaled)
            scaler         : If provided, inverse-transforms before computing metrics
        Returns:
            dict with MAPE, RMSE, MAE, directional_accuracy
        """
        y_pred_scaled = self.predict(X_test)

        if scaler is not None:
            # inverse transform: scaler expects shape (n, 1)
            y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        else:
            y_pred = y_pred_scaled
            y_true = y_test

        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = np.mean(np.abs(y_true - y_pred))

        # Directional accuracy: % times model correctly predicted up/down
        if len(y_true) > 1:
            actual_dir = np.sign(np.diff(y_true))
            pred_dir   = np.sign(np.diff(y_pred))
            dir_acc    = np.mean(actual_dir == pred_dir) * 100
        else:
            dir_acc = float("nan")

        metrics = {
            "MAPE (%)": round(mape, 4),
            "RMSE":     round(rmse, 4),
            "MAE":      round(mae, 4),
            "Directional Accuracy (%)": round(dir_acc, 2),
        }
        logger.info("Evaluation Metrics: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "ann_model.keras"))
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str, input_dim: int):
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        obj = cls(input_dim=input_dim, config=config)
        obj.model = keras.models.load_model(os.path.join(path, "ann_model.keras"))
        logger.info("Model loaded from %s", path)
        return obj
