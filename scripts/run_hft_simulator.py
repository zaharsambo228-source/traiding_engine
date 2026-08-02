import os

from loguru import logger
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.model.currencies import BTC, USDT
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CurrencyPair
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from src.strategy.hft_momentum import HFTMomentumConfig, HFTMomentumStrategy


def run_hft_simulation():
    catalog_path = "data/catalog"
    if not os.path.exists(catalog_path):
        logger.error(f"Каталог данных {catalog_path} не найден! Сначала запустите src/data/nautilus_converter.py")
        return

    logger.info("Загрузка DataCatalog...")
    catalog = ParquetDataCatalog(catalog_path)

    # Настраиваем движок симуляции
    engine_config = BacktestEngineConfig(trader_id="HFT-SIMULATOR-001")
    engine = BacktestEngine(config=engine_config)

    # Виртуальная биржа
    engine.add_venue(
        venue=Venue("BYBIT"),
        oms_type=OmsType.HEDGING,
        account_type=AccountType.MARGIN,
        base_currency=USDT,
        starting_balances=[Money(100.0, USDT)],
    )

    # Подключаем инструмент BTCUSDT
    instrument_id = InstrumentId.from_str("BTCUSDT.BYBIT")
    instrument = CurrencyPair(
        instrument_id=instrument_id,
        raw_symbol=Symbol("BTCUSDT"),
        base_currency=BTC,
        quote_currency=USDT,
        price_precision=8,
        size_precision=8,
        price_increment=Price.from_str("0.00000001"),
        size_increment=Quantity.from_str("0.00000001"),
        multiplier=Quantity.from_str("1.0"),
        lot_size=Quantity.from_str("0.00000001"),
        ts_event=0,
        ts_init=0,
    )
    engine.add_instrument(instrument)

    # Конфигурация стратегии
    strategy_config = HFTMomentumConfig(
        instrument_id=str(instrument_id), trade_size=0.01, max_position=0.05, seq_len=50
    )
    strategy = HFTMomentumStrategy(config=strategy_config)
    engine.add_strategy(strategy=strategy)

    # Данные для симуляции
    ticks = catalog.quote_ticks(instrument_id)
    engine.add_data(ticks)

    logger.info(f"Запуск симуляции HFT стратегии для {instrument_id}...")
    engine.run()

    # Результаты
    logger.success("\n=== СИМУЛЯЦИЯ ЗАВЕРШЕНА ===")
    logger.info("Смотрите логи выше для отчета по портфелю (PORTFOLIO PERFORMANCE).")


if __name__ == "__main__":
    run_hft_simulation()
