import glob
import os
import pickle

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from src.model.trainer import ModelTrainer
from src.model.transformer import TransformerTradingModel


class HFTDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        symbol: str = "BTC_USDT_USDT",
        seq_len: int = 50,
        future_steps: int = 10,
        threshold: float = 0.0001,
    ):
        self.seq_len = seq_len
        self.future_steps = future_steps
        self.threshold = threshold

        # 1. Читаем Parquet файлы только для конкретной монеты!
        parquet_files = sorted(glob.glob(f"{data_dir}/{symbol}_*.parquet"))
        if not parquet_files:
            raise ValueError(f"Нет файлов Parquet для {symbol} в {data_dir}")

        logger.info(f"Загрузка {len(parquet_files)} файлов...")

        # Pandas умеет сам читать папку с файлами, но мы считаем их по очереди для сортировки
        df_list = [pd.read_parquet(f) for f in parquet_files]
        self.df = pd.concat(df_list, ignore_index=True)
        self.df.sort_values("timestamp", inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        logger.info(f"Загружено {len(self.df)} тиков.")

        # 2. Выделяем фичи (микроструктура)
        feature_cols = [
            "bid_vol",
            "ask_vol",
            "spread",
            "ofi",
            "depth_ratio_50",
            "bid_depth_50",
            "ask_depth_50",
            "trade_imbalance",
        ]

        # Нормализация
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.df[feature_cols].values)

        # Сохраняем скейлер
        with open("hft_scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        # 3. Разметка таргетов (куда пойдет micro_price через future_steps тиков?)
        micro_prices = self.df["micro_price"].values
        self.labels = np.zeros(len(micro_prices), dtype=int)

        for i in range(len(micro_prices) - self.future_steps):
            current_p = micro_prices[i]
            future_p = micro_prices[i + self.future_steps]
            pct_change = (future_p - current_p) / current_p

            if pct_change > self.threshold:
                self.labels[i] = 1  # LONG
            elif pct_change < -self.threshold:
                self.labels[i] = 2  # SHORT
            else:
                self.labels[i] = 0  # FLAT

    def __len__(self):
        return len(self.features) - self.seq_len - self.future_steps

    def __getitem__(self, idx):
        # Берем окно из seq_len тиков
        x = self.features[idx : idx + self.seq_len]
        # Берем таргет в конце этого окна
        y = self.labels[idx + self.seq_len - 1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def main():
    data_dir = "data/hft"
    if not os.path.exists(data_dir):
        logger.error(f"Папка {data_dir} не найдена. Запустите сначала ws_collector.py")
        return

    logger.info("Подготовка HFT датасета...")
    # Обучаемся пока только на BTC, чтобы не смешивать тики разных монет
    dataset = HFTDataset(data_dir=data_dir, symbol="BTC_USDT_USDT", seq_len=50, future_steps=10)

    # DataLoader для батчинга
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)

    # У нас 7 микроструктурных фичей
    # Выгружаем весь датасет в numpy массивы для совместимости с ModelTrainer
    X_train = []
    y_train = []
    for i in range(len(dataset)):
        x, y = dataset[i]
        X_train.append(x.numpy())
        y_train.append(y.numpy())

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # 8 микроструктурных фичей
    model = TransformerTradingModel(input_dim=8, nhead=1, d_model=64, num_layers=2)

    logger.info("Запуск обучения HFT модели...")
    trainer = ModelTrainer(model, learning_rate=0.001)

    # MLflow tracking
    mlflow.set_experiment("HFT_Microstructure")
    with mlflow.start_run():
        mlflow.log_param("symbol", "BTC_USDT_USDT")
        mlflow.log_param("seq_len", 50)
        mlflow.log_param("future_steps", 10)
        mlflow.log_param("model", "TransformerTradingModel")
        mlflow.log_param("nhead", 1)
        mlflow.log_param("d_model", 64)
        mlflow.log_param("num_layers", 2)

        # Обучаем
        trainer.train(X_train, y_train, epochs=5)

        # Сохраняем модель локально
        torch.save(model.state_dict(), "hft_model.pth")
        logger.info("HFT модель успешно обучена и сохранена как hft_model.pth!")

        # Логируем модель в MLflow Registry
        mlflow.pytorch.log_model(model, "model", registered_model_name="HFT_Momentum_Model", input_example=X_train[:1])
        mlflow.log_artifact("hft_scaler.pkl")


if __name__ == "__main__":
    main()
