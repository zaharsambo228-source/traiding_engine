import numpy as np
import pandas as pd

from src.features.feature_engineering import calculate_indicators, extract_price_action


def create_mock_ohlcv(rows=60):
    """Creates a basic mock OHLCV dataframe for testing indicators."""
    np.random.seed(42)
    base_price = 100.0

    data = []
    current_time = 1600000000000
    for i in range(rows):
        change = np.random.normal(0, 0.5)
        close_price = base_price + change
        data.append(
            {
                "timestamp": current_time + (i * 60000),
                "open": base_price,
                "high": max(base_price, close_price) + 0.1,
                "low": min(base_price, close_price) - 0.1,
                "close": close_price,
                "volume": np.random.uniform(10, 100),
            }
        )
        base_price = close_price

    df = pd.DataFrame(data)
    return df


def test_calculate_indicators_empty():
    df = pd.DataFrame()
    result = calculate_indicators(df)
    assert result == {}


def test_calculate_indicators_valid():
    df = create_mock_ohlcv(60)
    result = calculate_indicators(df)

    # Assert all keys are present
    expected_keys = ["rsi", "macd", "macd_signal", "macd_diff", "bb_position", "atr"]
    for k in expected_keys:
        assert k in result

    # Assert values are floats and not nan
    for k, v in result.items():
        assert isinstance(v, float)
        assert not np.isnan(v)


def test_extract_price_action_valid():
    df = create_mock_ohlcv(20)
    result = extract_price_action(df, num_candles=5)

    # Assert correct number of features (5 candles * 3 features = 15 features)
    assert len(result) == 15

    # Assert values are floats
    for k, v in result.items():
        assert isinstance(v, float)
