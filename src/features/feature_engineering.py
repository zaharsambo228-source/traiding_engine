import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import ta


def calculate_indicators(df: pd.DataFrame) -> dict[str, float]:
    """Calculates technical indicators for a given dataframe slice."""
    if df.empty or len(df) < 50:
        return {}

    # Make sure data is sorted by timestamp
    df = df.sort_values("timestamp").copy()

    # Calculate RSI
    rsi = ta.momentum.RSIIndicator(close=df["close"], window=14)
    df["rsi"] = rsi.rsi()

    # Calculate MACD
    macd = ta.trend.MACD(close=df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    # Calculate Bollinger Bands
    bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_hband"] = bb.bollinger_hband()
    df["bb_lband"] = bb.bollinger_lband()

    # ATR
    atr = ta.volatility.AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["atr"] = atr.average_true_range()

    # Get the last row (most recent before the trade)
    last_row = df.iloc[-1]

    # Safely handle potential divisions by zero in bb_position
    bb_range = float(last_row["bb_hband"]) - float(last_row["bb_lband"]) + 1e-8
    bb_position = (float(last_row["close"]) - float(last_row["bb_lband"])) / bb_range

    return {
        "rsi": float(last_row["rsi"]) if not pd.isna(last_row["rsi"]) else 0.0,
        "macd": float(last_row["macd"]) if not pd.isna(last_row["macd"]) else 0.0,
        "macd_signal": float(last_row["macd_signal"]) if not pd.isna(last_row["macd_signal"]) else 0.0,
        "macd_diff": float(last_row["macd_diff"]) if not pd.isna(last_row["macd_diff"]) else 0.0,
        "bb_position": float(bb_position),
        "atr": float(last_row["atr"]) if not pd.isna(last_row["atr"]) else 0.0,
    }


def calculate_global_context(df_4h: pd.DataFrame) -> dict[str, float]:
    """Calculates higher timeframe context (trend, distance to SMA)."""
    if df_4h.empty or len(df_4h) < 20:
        return {}

    df_4h = df_4h.sort_values("timestamp").copy()

    sma20_series = ta.trend.SMAIndicator(close=df_4h["close"], window=20).sma_indicator()

    last_row = df_4h.iloc[-1]
    last_sma20 = float(sma20_series.iloc[-1])

    # Distance to SMA20 in percentage
    if pd.isna(last_sma20) or last_sma20 == 0:
        dist_sma20 = 0.0
    else:
        dist_sma20 = (float(last_row["close"]) - last_sma20) / last_sma20 * 100.0

    # Simple Trend (last close vs open 10 candles ago)
    if len(df_4h) >= 10:
        open_10 = float(df_4h.iloc[-10]["open"])
        if open_10 == 0:
            trend_10 = 0.0
        else:
            trend_10 = (float(last_row["close"]) - open_10) / open_10 * 100.0
    else:
        trend_10 = 0.0

    return {"4h_dist_sma20": float(dist_sma20), "4h_trend_10": float(trend_10)}


def extract_price_action(df_1m: pd.DataFrame, num_candles: int = 10) -> dict[str, float]:
    """Extracts raw price action normalized to the start of the window."""
    if df_1m.empty or len(df_1m) < num_candles:
        return {}

    # Get last N candles
    tail = df_1m.tail(num_candles)

    # Normalize relative to the first open of this sequence
    base_price = float(tail.iloc[0]["open"])
    if base_price == 0:
        return {}

    features: dict[str, float] = {}
    for i in range(num_candles):
        row = tail.iloc[i]
        prefix = f"pa_1m_{num_candles - i}"
        features[f"{prefix}_close"] = (float(row["close"]) - base_price) / base_price * 1000.0  # basis points
        features[f"{prefix}_high"] = (float(row["high"]) - base_price) / base_price * 1000.0
        features[f"{prefix}_low"] = (float(row["low"]) - base_price) / base_price * 1000.0

    return features


def process_dataset(filepath: str, out_dir: str) -> None:
    """Processes a raw parquet file and saves engineered features."""
    print(f"Feature engineering for {filepath}...")
    try:
        df_raw = pd.read_parquet(filepath)
    except Exception as e:
        print(f"Failed to read parquet {filepath}: {e}")
        return

    if df_raw.empty:
        return

    processed_records: list[dict[str, Any]] = []

    for _, row in df_raw.iterrows():
        try:
            df_1m = pd.DataFrame(json.loads(row["df_1m"]))
            df_5m = pd.DataFrame(json.loads(row["df_5m"]))
            df_4h = pd.DataFrame(json.loads(row["df_4h"]))
        except (ValueError, KeyError, TypeError):
            continue

        record: dict[str, Any] = {
            "trade_id": row.get("trade_id", ""),
            "symbol": row.get("symbol", ""),
            "side": row.get("side", ""),
            "pnl": float(row.get("pnl")) if row.get("pnl") is not None else 0.0,
            "label": int(row.get("label", 1)),
            "timestamp": int(row.get("open_time", 0)),
        }

        # Calculate features (make sure no leakage by using historical slices)
        ind_1m = calculate_indicators(df_1m)
        ind_5m = calculate_indicators(df_5m)
        ctx_4h = calculate_global_context(df_4h)
        pa_1m = extract_price_action(df_1m)

        if not ind_1m or not ind_5m or not ctx_4h or not pa_1m:
            continue

        # Add prefixes to indicators
        for k, v in ind_1m.items():
            record[f"1m_{k}"] = v
        for k, v in ind_5m.items():
            record[f"5m_{k}"] = v
        for k, v in ctx_4h.items():
            record[k] = v
        for k, v in pa_1m.items():
            record[k] = v

        processed_records.append(record)

    if processed_records:
        df_out = pd.DataFrame(processed_records)
        out_filepath = os.path.join(out_dir, f"{Path(filepath).stem}_features.parquet")
        df_out.to_parquet(out_filepath)
        print(f"Saved {len(processed_records)} feature rows to {out_filepath}")


def main() -> None:
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    in_dir = os.path.join(base_dir, "data", "enriched_samples")
    out_dir = os.path.join(base_dir, "data", "features")

    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(in_dir):
        print(f"Input directory {in_dir} not found.")
        return

    for filename in os.listdir(in_dir):
        if filename.endswith(".parquet"):
            filepath = os.path.join(in_dir, filename)
            process_dataset(filepath, out_dir)


if __name__ == "__main__":
    main()
