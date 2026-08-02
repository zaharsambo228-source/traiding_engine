import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands, DonchianChannel


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators using `ta` library.
    Total output features: 25 (21 base + 4 market-quality self-selecting features)
    """
    df = df.copy()

    # Trend
    df["ema_20"] = EMAIndicator(close=df["close"], window=20).ema_indicator()
    df["ema_50"] = EMAIndicator(close=df["close"], window=50).ema_indicator()
    df["ema_200"] = EMAIndicator(close=df["close"], window=200).ema_indicator()

    # Momentum
    df["rsi_14"] = RSIIndicator(close=df["close"], window=14).rsi()
    macd = MACD(close=df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    # Volatility
    bbands = BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_high"] = bbands.bollinger_hband()
    df["bb_low"] = bbands.bollinger_lband()
    df["bb_mid"] = bbands.bollinger_mavg()

    df["atr_14"] = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14).average_true_range()

    # Balance of Power (BOP)
    high_low_diff = df["high"] - df["low"]
    df["bop"] = (df["close"] - df["open"]) / high_low_diff.replace(0, 1e-8)

    # Donchian Channels (DC)
    dc = DonchianChannel(high=df["high"], low=df["low"], close=df["close"], window=20)
    df["dc_high"] = dc.donchian_channel_hband()
    df["dc_low"] = dc.donchian_channel_lband()
    df["dc_mid"] = dc.donchian_channel_mband()

    # ================================================================
    # НОВЫЕ ПРИЗНАКИ: "КАЧЕСТВО МОНЕТЫ" (4 штуки, input_dim: 21 → 25)
    # Модель учится сама: низкое natr/vol_ratio → выдаёт FLAT
    # ================================================================

    atr_safe = df["atr_14"].replace(0, 1e-8)
    close_safe = df["close"].replace(0, 1e-8)

    # 1. NATR — нормализованная волатильность (ATR / цена * 100)
    #    BTC: natr≈0.05%   |   горячий альткоин: natr≈3-5%
    df["natr"] = (atr_safe / close_safe) * 100

    # 2. VOL_RATIO — объём свечи / скользящее среднее объёма за 20 свечей
    #    vol_ratio > 2.5 → аномальный объём → потенциальный импульс
    vol_mean_20 = df["volume"].rolling(window=20, min_periods=5).mean().replace(0, 1e-8)
    df["vol_ratio"] = df["volume"] / vol_mean_20

    # 3. VOL_ZSCORE — z-score объёма относительно 20-периодного окна
    #    zscore > 2.0 = статистически аномальное событие
    vol_std_20 = df["volume"].rolling(window=20, min_periods=5).std().replace(0, 1e-8)
    df["vol_zscore"] = (df["volume"] - vol_mean_20) / vol_std_20

    # 4. CANDLE_STRENGTH — направленность свечи, нормализованная на ATR
    #    +1.0 → сильная бычья свеча    -1.0 → сильная медвежья
    df["candle_strength"] = (df["close"] - df["open"]) / atr_safe

    return df
