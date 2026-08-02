import joblib
import numpy as np
import torch
from loguru import logger


class ModelInference:
    def __init__(self, model_class, model_path: str, scaler_path: str, input_dim: int, **model_kwargs):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # Initialize model architecture
        self.model = model_class(input_dim=input_dim, **model_kwargs).to(self.device)

        # Load weights
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            logger.info(f"Loaded model from {model_path}")

            # Load scaler (e.g., StandardScaler from scikit-learn)
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from {scaler_path}")
        except Exception as e:
            logger.error(f"Failed to load model or scaler: {e}")
            raise

    def predict(self, sequence: np.ndarray) -> (int, float):
        """
        Predict the class and return confidence.
        sequence shape: (sequence_length, input_dim)
        Returns: (predicted_class, confidence_score)
        """
        with torch.no_grad():
            # Scale features
            scaled_seq = self.scaler.transform(sequence)

            # Convert to tensor and add batch dimension
            x_tensor = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Forward pass
            output = self.model(x_tensor)

            # Apply softmax for probabilities
            probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()

            predicted_class = np.argmax(probs)
            confidence = probs[predicted_class]

            return int(predicted_class), float(confidence)
