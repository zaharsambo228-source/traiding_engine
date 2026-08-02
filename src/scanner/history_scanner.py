import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from typing import Any
import argparse

import ccxt
from catboost import CatBoostClassifier
import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore")

# Add src to path so we can import features
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(base_dir)

from src.features.feature_engineering import calculate_global_context, calculate_indicators, extract_price_action

exchange = ccxt.bybit(
    {
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",
        },
    }
)


def format_symbol_for_bybit(symbol: str) -> str:
    """Formats a basic symbol like BTCUSDT to Bybit's internal representation."""
    if symbol.endswith("USDT") and not symbol.endswith(":USDT"):
        base = symbol[:-4]
        return f"{base}/USDT:USDT"
    return symbol


def fetch_historical_data_bulk(symbol: str, timeframe: str, since_ms: int, until_ms: int) -> pd.DataFrame:
    """Fetches historical data in chunks up to `until_ms`."""
    logger.info(f"Fetching {timeframe} history for {symbol}...")
    all_ohlcv: list[list[Any]] = []
    current_since = since_ms

    while current_since < until_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
            if not ohlcv:
                break

            all_ohlcv.extend(ohlcv)
            last_ts = ohlcv[-1][0]
            if last_ts <= current_since:
                break  # prevent infinite loop
            current_since = last_ts + 1

            time.sleep(0.1)  # Respect rate limits
        except ccxt.NetworkError as e:
            logger.warning(f"Network error fetching {symbol} {timeframe}: {e}. Retrying in 5s...")
            time.sleep(5)
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error for {symbol} {timeframe}: {e}. Skipping further fetch.")
            break
        except Exception as e:
            logger.error(f"Unexpected error fetching {symbol} {timeframe}: {e}")
            break

    if not all_ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def simulate_forward_trade(current_ts: int, entry_price: float, df_1m: pd.DataFrame, sl_pct: float, tp_pct: float, max_time_ms: int) -> tuple[float, float]:
    """
    Simulates a trade going forward in time without data leakage.
    Returns (long_pnl, short_pnl).
    """
    future_df = df_1m[(df_1m["timestamp"] > current_ts) & (df_1m["timestamp"] <= current_ts + max_time_ms)]
    
    long_pnl = 0.0
    short_pnl = 0.0
    
    long_sl_price = entry_price * (1 - sl_pct)
    long_tp_price = entry_price * (1 + tp_pct)
    short_sl_price = entry_price * (1 + sl_pct)
    short_tp_price = entry_price * (1 - tp_pct)
    
    long_active = True
    short_active = True
    
    for _, row in future_df.iterrows():
        if not long_active and not short_active:
            break
            
        high = row["high"]
        low = row["low"]
        
        # Check Long
        if long_active:
            if low <= long_sl_price:
                long_pnl = -1.0 # Hit SL (normalized to 1R risk)
                long_active = False
            elif high >= long_tp_price:
                long_pnl = 3.0 # Hit TP (3R reward)
                long_active = False
                
        # Check Short
        if short_active:
            if high >= short_sl_price:
                short_pnl = -1.0
                short_active = False
            elif low <= short_tp_price:
                short_pnl = 3.0
                short_active = False
                
    return long_pnl, short_pnl


def scan_history(
    symbol_raw: str, model: Any, feature_cols: list[str], days_back: int = 30, threshold: float = 0.85,
    sl_pct: float = 0.01, tp_pct: float = 0.03
) -> list[dict[str, Any]]:
    """Scans historical data for high probability setups and simulates forward PnL."""
    symbol_ccxt = format_symbol_for_bybit(symbol_raw)
    logger.info(f"\n--- Starting Historical Scan for {symbol_raw} ({days_back} days) ---")

    until_ms = int(datetime.now().timestamp() * 1000)
    # 4H data needs more history for indicators
    since_ms_4h = int((datetime.now() - timedelta(days=days_back + 30)).timestamp() * 1000)
    since_ms_1m = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)

    df_1m = fetch_historical_data_bulk(symbol_ccxt, "1m", since_ms_1m, until_ms)
    df_5m = fetch_historical_data_bulk(symbol_ccxt, "5m", since_ms_1m, until_ms)
    df_4h = fetch_historical_data_bulk(symbol_ccxt, "4h", since_ms_4h, until_ms)

    if df_1m.empty or df_5m.empty or df_4h.empty:
        logger.warning(f"Not enough data for {symbol_raw}")
        return []

    logger.info(f"Loaded {len(df_1m)} 1m candles, {len(df_5m)} 5m candles, {len(df_4h)} 4h candles.")

    matches: list[dict[str, Any]] = []
    lookback = 100  # Candles required for indicator calculation
    step = 5  # Step in minutes to slide window

    # Pre-calculate indicators for the whole dataframe if possible, but to be strictly safe from leakage 
    # and mimic production exactly, we calculate window-by-window, or use rolling.
    # We will stick to the window-by-window for exact parity with the fetcher, even if slower.
    
    count_scanned = 0
    # Stop scanning 24h before the end of data to allow for forward simulation
    max_forward_ms = 24 * 60 * 60 * 1000 
    
    for i in range(lookback, len(df_1m), step):
        current_ts = df_1m.iloc[i]["timestamp"]
        
        # Don't scan the very last 24h because we can't fully simulate forward PnL
        if current_ts > until_ms - max_forward_ms:
            break
            
        count_scanned += 1
        if count_scanned % 1000 == 0:
            logger.info(f"Scanned {count_scanned} points for {symbol_raw}...")

        slice_1m = df_1m.iloc[i - lookback : i + 1]
        
        # Get 5m and 4h slices <= current_ts
        slice_5m = df_5m[df_5m["timestamp"] <= current_ts].tail(lookback)
        if len(slice_5m) < 20:
            continue
        slice_4h = df_4h[df_4h["timestamp"] <= current_ts].tail(lookback)
        if len(slice_4h) < 20:
            continue

        ind_1m = calculate_indicators(slice_1m.copy())
        ind_5m = calculate_indicators(slice_5m.copy())
        ctx_4h = calculate_global_context(slice_4h.copy())
        pa_1m = extract_price_action(slice_1m.copy())

        if not ind_1m or not ind_5m or not ctx_4h or not pa_1m:
            continue

        record: dict[str, Any] = {}
        for k, v in ind_1m.items():
            record[f"1m_{k}"] = v
        for k, v in ind_5m.items():
            record[f"5m_{k}"] = v
        for k, v in ctx_4h.items():
            record[k] = v
        for k, v in pa_1m.items():
            record[k] = v

        df_features = pd.DataFrame([record])
        # Ensure all columns expected by the model exist
        for col in feature_cols:
            if col not in df_features.columns:
                df_features[col] = 0.0

        X = df_features[feature_cols]
        prob = model.predict_proba(X)[0][1] # Probability of class 1

        if prob > threshold:
            entry_price = slice_1m.iloc[-1]["close"]
            dt_str = pd.to_datetime(current_ts, unit='ms')
            
            logger.info(f"[{dt_str}] Setup Found! Prob: {prob:.4f}")
            
            long_pnl, short_pnl = simulate_forward_trade(
                current_ts=current_ts,
                entry_price=entry_price,
                df_1m=df_1m,
                sl_pct=sl_pct,
                tp_pct=tp_pct,
                max_time_ms=max_forward_ms
            )
            
            logger.info(f"  -> Simulated PnL (L/S): {long_pnl} / {short_pnl}")
            
            match_data = {
                "symbol": symbol_raw,
                "timestamp": current_ts,
                "datetime": str(dt_str),
                "prob": prob,
                "close_price": entry_price,
                "long_pnl": long_pnl,
                "short_pnl": short_pnl
            }
            # Merge features into match_data
            for col in feature_cols:
                match_data[col] = record.get(col, 0.0)
                
            matches.append(match_data)

    logger.info(f"Finished {symbol_raw}. Found {len(matches)} setup points.")
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan history and generate RL dataset")
    parser.add_argument("--symbol", type=str, default="WIFUSDT", help="Symbol to scan (e.g. WIFUSDT)")
    parser.add_argument("--days", type=int, default=30, help="Days of history to scan")
    parser.add_argument("--threshold", type=float, default=0.6, help="CatBoost probability threshold")
    args = parser.parse_args()

    models_dir = os.path.join(base_dir, "models")
    model_path = os.path.join(models_dir, "setup_classifier.cbm")

    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}. Train it first!")
        return

    logger.info(f"Loading CatBoost model from {model_path}...")
    model = CatBoostClassifier()
    model.load_model(model_path)
    feature_cols = model.feature_names_

    all_matches: list[dict[str, Any]] = []
    
    # We can scan just one symbol or multiple, depending on args.
    # To keep it safe and parallelizable, we scan one symbol per run if specified.
    sym = args.symbol
    
    matches = scan_history(
        symbol_raw=sym, 
        model=model, 
        feature_cols=feature_cols, 
        days_back=args.days, 
        threshold=args.threshold,
        sl_pct=0.01, # 1% sl
        tp_pct=0.03  # 3% tp
    )
    all_matches.extend(matches)

    if all_matches:
        out_dir = os.path.join(base_dir, "data", "rl_dataset")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = os.path.join(out_dir, f"sim_{sym}_{ts}.parquet")

        pd.DataFrame(all_matches).to_parquet(out_file)
        logger.success(f"Saved {len(all_matches)} simulated setups to {out_file}!")
    else:
        logger.warning("No matches found above threshold in the historical scan.")


if __name__ == "__main__":
    main()
