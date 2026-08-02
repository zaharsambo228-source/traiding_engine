import os

from loguru import logger
from nautilus_trader.adapters.bybit.config import BybitDataClientConfig, BybitExecClientConfig, BybitOmsConfig
from nautilus_trader.adapters.bybit.factories import BybitLiveAdapterFactory
from nautilus_trader.config import LiveExecEngineConfig
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.identifiers import InstrumentId

from src.strategy.hft_momentum import HFTMomentumConfig, HFTMomentumStrategy


def run_live_bot():
    api_key = os.environ.get("BYBIT_API_KEY")
    api_secret = os.environ.get("BYBIT_API_SECRET")

    if not api_key or not api_secret:
        logger.error("Для Live торговли необходимо установить переменные окружения BYBIT_API_KEY и BYBIT_API_SECRET")
        return

    logger.info("Инициализация Live узла NautilusTrader...")

    # Конфигурация узла
    node_config = LiveExecEngineConfig(log_level="INFO", bypass_logging=False)
    node = TradingNode(config=node_config)

    # Конфигурация адаптера Bybit
    adapter_factory = BybitLiveAdapterFactory()

    # 1. Data Client (WebSocket L2)
    data_client_config = BybitDataClientConfig(
        api_key=api_key,
        api_secret=api_secret,
        usdt_linear_public_ws_url="wss://stream.bybit.com/v5/public/linear",
    )
    node.add_data_client(adapter_factory.create_data_client(data_client_config))

    # 2. Execution Client (Routing orders)
    exec_client_config = BybitExecClientConfig(
        api_key=api_key,
        api_secret=api_secret,
        usdt_linear_private_ws_url="wss://stream.bybit.com/v5/private",
    )
    node.add_exec_client(adapter_factory.create_exec_client(exec_client_config))

    # 3. Order Management System (OMS)
    oms_config = BybitOmsConfig(
        api_key=api_key,
        api_secret=api_secret,
    )
    node.add_oms(adapter_factory.create_oms(oms_config))

    node.build()

    # Добавляем стратегию (использует модель из MLflow, которую мы сохраняли локально)
    instrument_id = InstrumentId.from_str("BTCUSDT.BYBIT")

    strategy_config = HFTMomentumConfig(
        instrument_id=str(instrument_id), trade_size=0.01, max_position=0.05, seq_len=50
    )
    strategy = HFTMomentumStrategy(config=strategy_config)
    node.add_strategy(strategy=strategy)

    logger.warning(">>> ВНИМАНИЕ: БОТ ЗАПУСКАЕТСЯ В РЕЖИМЕ РЕАЛЬНОЙ ТОРГОВЛИ <<<")
    logger.info("Подключение к Bybit WebSocket и запуск цикла...")

    node.start()

    try:
        # Держим главный поток живым, пока бот торгует в фоне
        while True:
            import time

            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Остановка узла...")
        node.stop()


if __name__ == "__main__":
    run_live_bot()
