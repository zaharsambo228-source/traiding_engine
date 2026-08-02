import collections
import pickle

import numpy as np
import torch
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy

from src.model.transformer import TransformerTradingModel


class HFTMomentumConfig(StrategyConfig, frozen=True):
    instrument_id: str
    trade_size: float = 0.01
    max_position: float = 0.05
    seq_len: int = 50
    entry_threshold: float = 0.40


class HFTMomentumStrategy(Strategy):
    def __init__(self, config: HFTMomentumConfig):
        super().__init__(config)
        self.instrument = None

        # Буфер для тиков (храним последние seq_len тиков)
        self.ticks_buffer = collections.deque(maxlen=config.seq_len)

        # Переменные для расчета OFI
        self.prev_bid_price = 0.0
        self.prev_ask_price = 0.0
        self.prev_bid_vol = 0.0
        self.prev_ask_vol = 0.0

        # Буфер фичей (каждый элемент - массив из 7 фичей)
        self.features_buffer = collections.deque(maxlen=config.seq_len)

        self.model = None
        self.scaler = None

    def on_start(self):
        self.instrument = self.cache.instrument(InstrumentId.from_str(self.config.instrument_id))
        if self.instrument is None:
            self.log.error(f"Инструмент {self.config.instrument_id} не найден.")
            return

        # Подписка на тики
        self.subscribe_quote_ticks(self.instrument.id)

        # Загрузка нейросети
        self.log.info("Загрузка HFT модели и скейлера...")
        self.model = TransformerTradingModel(input_dim=8, nhead=1, d_model=64, num_layers=2)
        try:
            self.model.load_state_dict(torch.load("hft_model.pth", map_location=torch.device("cpu")))
            self.model.eval()

            with open("hft_scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            self.log.info("Модель успешно загружена!")
        except Exception as e:
            self.log.error(f"Ошибка загрузки модели: {e}")

    def _calc_ofi(self, bid_price, ask_price, bid_vol, ask_vol):
        # Calculation of Order Flow Imbalance (OFI)
        if bid_price >= self.prev_bid_price:
            bid_diff = bid_vol
        else:
            bid_diff = -bid_vol

        if ask_price <= self.prev_ask_price:
            ask_diff = ask_vol
        else:
            ask_diff = -ask_vol

        ofi = bid_diff - ask_diff

        self.prev_bid_price = bid_price
        self.prev_ask_price = ask_price
        self.prev_bid_vol = bid_vol
        self.prev_ask_vol = ask_vol

        return ofi

    def on_quote_tick(self, tick: QuoteTick):
        bid_p = tick.bid_price.as_double()
        ask_p = tick.ask_price.as_double()
        bid_v = tick.bid_size.as_double()
        ask_v = tick.ask_size.as_double()

        spread = ask_p - bid_p
        ofi = self._calc_ofi(bid_p, ask_p, bid_v, ask_v)

        # Упрощенные фичи глубины
        depth_ratio_50 = bid_v / (ask_v + 1e-8)
        bid_depth_50 = bid_v * 10  # mock as we don't have full L2 in QuoteTick
        ask_depth_50 = ask_v * 10

        # mock trade imbalance as QuoteTick doesn't have trades
        trade_imbalance = 0.0

        feature_vector = np.array(
            [bid_v, ask_v, spread, ofi, depth_ratio_50, bid_depth_50, ask_depth_50, trade_imbalance]
        )
        self.features_buffer.append(feature_vector)

        # Ждем, пока буфер заполнится
        if len(self.features_buffer) < self.config.seq_len:
            return

        # 1. Нормализация
        raw_seq = np.array(self.features_buffer)
        scaled_seq = self.scaler.transform(raw_seq)

        # 2. Инференс (временно хардкодим symbol_id=0 для BTC)
        with torch.no_grad():
            x_tensor = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0)  # добавляем batch_size=1
            sym_id = torch.tensor([0], dtype=torch.long)
            output = self.model(x_tensor, symbol_ids=sym_id)

            # Применяем softmax для получения вероятностей
            probs = torch.softmax(output, dim=1).squeeze(0)  # [prob_FLAT, prob_LONG, prob_SHORT]

            prob_long = probs[1].item()
            prob_short = probs[2].item()

            pred = 0
            if prob_long > self.config.entry_threshold and prob_long > prob_short:
                pred = 1
            elif prob_short > self.config.entry_threshold and prob_short > prob_long:
                pred = 2
        # 3. Торговая логика
        # pred: 0=FLAT, 1=LONG, 2=SHORT
        pos_list = self.cache.positions(instrument_id=self.instrument.id)
        pos = pos_list[0] if pos_list else None
        current_qty = pos.quantity.as_double() if pos else 0.0

        qty = Quantity.from_str(str(self.config.trade_size))

        if pred == 1:
            if current_qty < self.config.max_position:
                self.submit_market_order(instrument_id=self.instrument.id, order_side=OrderSide.BUY, quantity=qty)
        elif pred == 2:
            if current_qty > -self.config.max_position:
                self.submit_market_order(instrument_id=self.instrument.id, order_side=OrderSide.SELL, quantity=qty)
