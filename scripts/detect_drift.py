import glob
import os

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from loguru import logger


def get_recent_files(data_dir: str, n: int = 5):
    """Возвращает n самых новых parquet файлов"""
    files = sorted(glob.glob(f"{data_dir}/*.parquet"), key=os.path.getmtime, reverse=True)
    return files[:n]


def get_oldest_files(data_dir: str, n: int = 5):
    """Возвращает n самых старых parquet файлов (baseline)"""
    files = sorted(glob.glob(f"{data_dir}/*.parquet"), key=os.path.getmtime, reverse=False)
    return files[:n]


def main():
    data_dir = "data/hft"

    baseline_files = get_oldest_files(data_dir, 3)
    recent_files = get_recent_files(data_dir, 3)

    if not baseline_files or not recent_files:
        logger.error("Недостаточно данных для сравнения Drift'а")
        return

    logger.info("Загрузка Baseline (старых) данных...")
    df_base = pd.concat([pd.read_parquet(f) for f in baseline_files])

    logger.info("Загрузка Recent (новых) данных...")
    df_recent = pd.concat([pd.read_parquet(f) for f in recent_files])

    # Сравниваем только числовые микроструктурные фичи
    features = ["bid_vol", "ask_vol", "spread", "ofi", "depth_ratio_50", "trade_imbalance"]
    # Проверяем, есть ли фичи в датафрейме (старые файлы могли не иметь trade_imbalance)
    features = [f for f in features if f in df_base.columns and f in df_recent.columns]

    df_base = df_base[features]
    df_recent = df_recent[features]

    logger.info("Анализ Data Drift с помощью Evidently AI...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=df_base, current_data=df_recent)

    report_path = "drift_report.html"
    report.save_html(report_path)
    logger.success(f"Отчет сохранен в {report_path}! Откройте его в браузере.")

    # Проверка на наличие критического дрифта
    drift_result = report.as_dict()
    dataset_drift = drift_result["metrics"][0]["result"]["dataset_drift"]

    if dataset_drift:
        logger.warning("ОБНАРУЖЕН DATA DRIFT! Волатильность стакана изменилась. Требуется переобучение модели.")
        logger.info("Автоматический запуск скрипта переобучения...")
        # В проде здесь будет триггер Airflow или os.system("python scripts/train_hft.py")
    else:
        logger.info("Рыночный режим стабилен. Data Drift не обнаружен.")


if __name__ == "__main__":
    main()
