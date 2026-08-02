import numpy as np
import pandas as pd

from .indicators import add_indicators


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features for ML model.
    """
    # 1. Add base indicators
    df = add_indicators(df)

    # 2. Add derived features (log returns, normalized momentum)
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["vol_change"] = df["volume"] / df["volume"].shift(1)

    # 3. Add target variable (Future 5m return)
    # E.g., looking 5 periods ahead
    df["target_5"] = np.log(df["close"].shift(-5) / df["close"])

    # Note: For classification, target could be:
    # df['target_class'] = np.where(df['target_5'] > 0.002, 1, np.where(df['target_5'] < -0.002, -1, 0))

    # Drop NaNs created by shifts and indicators
    df.dropna(inplace=True)
    return df
