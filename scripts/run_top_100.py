import ccxt
import os
from loguru import logger

def main():
    logger.info("Подключаемся к Bybit для получения ТОП-100 монет по объему...")
    exchange = ccxt.bybit({'options': {'defaultType': 'swap'}})
    exchange.load_markets()
    tickers = exchange.fetch_tickers()

    markets_vol = []
    for sym, ticker in tickers.items():
        # Берем только бессрочные USDT фьючерсы
        if sym.endswith("/USDT:USDT"):
            vol = ticker.get('quoteVolume', 0)
            if vol is not None:
                markets_vol.append((sym, vol))

    # Сортируем по объему торгов по убыванию и берем ТОП-100
    markets_vol.sort(key=lambda x: x[1], reverse=True)
    top_100 = [x[0].replace("/USDT:USDT", "USDT") for x in markets_vol[:100]]

    logger.info(f"Отобрано {len(top_100)} самых популярных монет: {', '.join(top_100[:10])}...")

    # Запускаем наш history_scanner для каждой монеты
    for idx, coin in enumerate(top_100, 1):
        logger.info(f"\n=============================================")
        logger.info(f" ЗАПУСК {idx}/100: {coin}")
        logger.info(f"=============================================")
        
        # Запускаем сбор за 3 года (1095 дней). Порог (threshold) у нас по умолчанию 0.6.
        cmd = f"python -m src.scanner.history_scanner --symbol {coin} --days 1095"
        
        # Выполняем команду
        exit_code = os.system(cmd)
        
        if exit_code != 0:
            logger.error(f"Произошла ошибка при сканировании {coin}. Идем к следующей монете...")

if __name__ == "__main__":
    main()
