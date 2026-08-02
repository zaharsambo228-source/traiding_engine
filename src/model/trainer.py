import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset


class ModelTrainer:
    def __init__(self, model: nn.Module, learning_rate: float = 0.001):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        logger.info(f"Using device: {self.device}")

        self.model = model.to(self.device)
        # criterion будет создан в train() после подсчета весов классов
        self.criterion = None
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 50, batch_size: int = 64):
        logger.info("Starting model training...")

        # Автоматический расчет весов классов: чем реже класс — тем выше его вес
        unique_classes = np.unique(y_train)
        class_weights = compute_class_weight("balanced", classes=unique_classes, y=y_train)
        # Если какого-то класса нет в выборке — заполняем вес 1.0 для полного списка [0, 1, 2]
        full_weights = np.ones(3)
        for cls, w in zip(unique_classes, class_weights):
            full_weights[int(cls)] = min(w, 5.0)  # Capping the weight at 5.0
        weight_tensor = torch.tensor(full_weights, dtype=torch.float32).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        logger.info(
            f"Веса классов: FLAT={full_weights[0]:.2f}, LONG={full_weights[1]:.2f}, SHORT={full_weights[2]:.2f}"
        )

        # Convert to tensors
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

        logger.info("Training completed.")

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
