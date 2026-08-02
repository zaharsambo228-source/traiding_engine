import glob
import os
import matplotlib.pyplot as plt
import pandas as pd
from catboost import CatBoostClassifier, Pool
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import mlflow

def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    features_dir = os.path.join(base_dir, "data", "features")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    parquet_files = glob.glob(f"{features_dir}/*.parquet")
    if not parquet_files:
        logger.error(f"Не найдены файлы с фичами в {features_dir}. Запустите сначала feature_engineering.py")
        return

    logger.info(f"Загрузка {len(parquet_files)} файлов с фичами...")
    df_list = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(df_list, ignore_index=True)
    logger.info(f"Всего загружено {len(df)} сэмплов (сделок и шума).")

    # Исключаем колонки, которые не являются фичами
    exclude_cols = ["trade_id", "symbol", "side", "timestamp", "pnl", "label"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Убираем строки с NaN
    df = df.dropna(subset=feature_cols + ["label"])

    X = df[feature_cols]
    y = df["label"].astype(int)

    logger.info(f"Используем {len(feature_cols)} фичей.")
    logger.info(f"Баланс классов: \n{y.value_counts(normalize=True)}")

    # Разбиваем на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Инициализация модели CatBoost
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        eval_metric="AUC",
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50
    )

    train_pool = Pool(X_train, y_train)
    test_pool = Pool(X_test, y_test)

    mlflow.set_experiment("Setup_Classifier_Boosting")
    with mlflow.start_run():
        logger.info("Начало обучения CatBoost...")
        model.fit(train_pool, eval_set=test_pool)

        # Оценка
        preds = model.predict(X_test)
        preds_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, preds_proba)

        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"ROC-AUC: {auc:.4f}")
        logger.info(f"\n{classification_report(y_test, preds)}")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_param("iterations", model.get_param("iterations"))
        mlflow.log_param("depth", model.get_param("depth"))

        # Сохранение модели
        model_path = os.path.join(models_dir, "setup_classifier.cbm")
        model.save_model(model_path)
        logger.info(f"Модель сохранена в {model_path}")

        # Feature Importance
        feature_importances = model.get_feature_importance(train_pool)
        fi_df = pd.DataFrame({"feature": feature_cols, "importance": feature_importances})
        fi_df = fi_df.sort_values(by="importance", ascending=False).head(20)

        plt.figure(figsize=(10, 8))
        plt.barh(fi_df["feature"], fi_df["importance"])
        plt.gca().invert_yaxis()
        plt.title("Топ-20 самых важных фичей (CatBoost)")
        
        fi_plot_path = os.path.join(models_dir, "feature_importance.png")
        plt.savefig(fi_plot_path, bbox_inches="tight")
        mlflow.log_artifact(fi_plot_path)
        mlflow.catboost.log_model(model, "catboost_model")
        
        logger.info("График важности фичей сохранен!")


if __name__ == "__main__":
    main()
