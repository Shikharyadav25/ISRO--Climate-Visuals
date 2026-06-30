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
        self.load_convlstm_model()

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
            logger.info(" Rainfall LSTM model loaded")

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

            logger.info(" Temperature models loaded")

            # ==========================
            # Load Scalers
            # ==========================
            with open(self.model_dir / "rainfall_scaler.pkl", "rb") as f:
                self.scalers["rainfall"] = pickle.load(f)

            with open(self.model_dir / "temp_feature_scaler.pkl", "rb") as f:
                self.scalers["temp_features"] = pickle.load(f)

            logger.info(" All scalers loaded")

        except Exception as e:
            logger.exception("Failed to load models.")
            raise e

    def load_convlstm_model(self):
        """Load trained PyTorch ConvLSTM models for 2D spatial grid prediction"""
        try:
            import torch
            from src.models.pytorch_convlstm import SpatioTemporalConvLSTM
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Rainfall ConvLSTM
            checkpoint_path = Path(__file__).resolve().parent.parent / "checkpoints" / "climate_twin_convlstm_final.pth"
            if checkpoint_path.exists():
                model = SpatioTemporalConvLSTM(input_dim=1, hidden_dim=[64, 32], kernel_size=(3, 3), num_layers=2)
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
                model.to(device)
                model.eval()
                self.models["convlstm"] = model
                logger.info(" PyTorch Rainfall ConvLSTM model loaded successfully")
            else:
                self.models["convlstm"] = None
                logger.warning("PyTorch Rainfall ConvLSTM checkpoint not found. Will use autoregressive proxy.")
                
            # Temperature ConvLSTM
            temp_checkpoint_path = Path(__file__).resolve().parent.parent / "checkpoints" / "climate_twin_convlstm_temp.pth"
            if temp_checkpoint_path.exists():
                temp_model = SpatioTemporalConvLSTM(input_dim=1, hidden_dim=[64, 32], kernel_size=(3, 3), num_layers=2)
                temp_model.load_state_dict(torch.load(temp_checkpoint_path, map_location=device))
                temp_model.to(device)
                temp_model.eval()
                self.models["convlstm_temp"] = temp_model
                logger.info(" PyTorch Temperature ConvLSTM model loaded successfully")
            else:
                self.models["convlstm_temp"] = None
                logger.warning("PyTorch Temperature ConvLSTM checkpoint not found. Will use autoregressive proxy.")
        except Exception as e:
            logger.exception("Failed to load PyTorch ConvLSTM models.")
            self.models["convlstm"] = None
            self.models["convlstm_temp"] = None

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