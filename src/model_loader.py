import tensorflow as tf
import pickle
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self, model_dir=None):
        """
        Initialize model loader.

        If model_dir is not provided, automatically locate the project's
        models/ folder regardless of the current working directory.
        """
        if model_dir is None:
            self.model_dir = Path(__file__).resolve().parent.parent / "models"
        else:
            self.model_dir = Path(model_dir)

        self.models = {}
        self.scalers = {}

        self.load_all_models()

    def load_all_models(self):
        """Load all trained models and scalers"""

        try:
            # ==========================
            # Rainfall LSTM
            # ==========================
            self.models["rainfall_lstm"] = tf.keras.models.load_model(
                self.model_dir / "rainfall_lstm_model.h5",
                compile=False
            )
            logger.info("✓ Rainfall LSTM model loaded")

            # ==========================
            # Temperature Models
            # ==========================
            self.models["max_temp"] = tf.keras.models.load_model(
                self.model_dir / "max_temp_model.h5",
                compile=False
            )

            self.models["min_temp"] = tf.keras.models.load_model(
                self.model_dir / "min_temp_model.h5",
                compile=False
            )

            logger.info("✓ Temperature models loaded")

            # ==========================
            # Load Scalers
            # ==========================
            with open(self.model_dir / "rainfall_scaler.pkl", "rb") as f:
                self.scalers["rainfall"] = pickle.load(f)

            with open(self.model_dir / "temp_feature_scaler.pkl", "rb") as f:
                self.scalers["temp_features"] = pickle.load(f)

            logger.info("✓ All scalers loaded")

        except Exception as e:
            logger.exception("Failed to load models.")
            raise e

    def predict_rainfall(self, sequence):
        """
        Predict rainfall from a 30-day rainfall sequence.
        """
        prediction = self.models["rainfall_lstm"].predict(sequence, verbose=0)
        prediction = self.scalers["rainfall"].inverse_transform(prediction)
        return prediction

    def predict_temperature(self, features):
        """
        Predict max and min temperatures.
        """
        features_scaled = self.scalers["temp_features"].transform(features)

        max_temp = self.models["max_temp"].predict(
            features_scaled,
            verbose=0
        )

        min_temp = self.models["min_temp"].predict(
            features_scaled,
            verbose=0
        )

        return max_temp, min_temp