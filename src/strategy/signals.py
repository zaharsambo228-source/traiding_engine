from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class SignalType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"  # Close positions


@dataclass
class TradeSignal:
    symbol: str
    timestamp: datetime
    type: SignalType
    confidence: float  # 0.0 to 1.0 from ML model
    price: float
    stop_loss: float
    take_profit: float
