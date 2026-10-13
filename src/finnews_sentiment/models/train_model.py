import json
from pathlib import Path
import pandas as pd
import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, f1_score

from sklearn.model_selection import train_test_split
from finnews_sentiment.features.preprocess import ID2LABEL, preprocess_text
import joblib


def load_config(path: Path) -> dict:
    """Читает JSON-конфиг обучения."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_baseline_pipeline() -> Pipeline:
    """Чистка текста + TF-IDF (слова и пары слов) + логистическая регрессия."""
    tfidf = TfidfVectorizer(preprocessor=preprocess_text, max_features=5000, ngram_range=(1, 2))
    clf = LogisticRegression(max_iter=1000)
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def split_data(df: pd.DataFrame, cfg: dict) -> tuple:
    """Стратифицированное разбиение на train/test по параметрам из конфига."""
    return train_test_split(
        df[cfg["text_col"]], df[cfg["target_col"]],
        test_size=cfg["test_size"], random_state=cfg["random_state"],
        stratify=df[cfg["target_col"]],
    )


def evaluate(pipe: Pipeline, X_test: pd.Series, y_test: pd.Series) -> dict:
    """Печатает classification_report и возвращает ключевые метрики."""
    y_pred = pipe.predict(X_test)
    target_names = [ID2LABEL[i] for i in sorted(ID2LABEL)]
    print(classification_report(y_test, y_pred, target_names=target_names, digits=3))
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "macro_f1": round(f1_score(y_test, y_pred, average="macro"), 4),
    }


def main(config_path: str) -> None:
    cfg = load_config(Path(config_path))
    df = pd.read_csv(cfg["input_path"])
    X_train, X_test, y_train, y_test = split_data(df, cfg)

    pipe = build_baseline_pipeline()
    pipe.fit(X_train, y_train)
    metrics = evaluate(pipe, X_test, y_test)
    print(f"Macro F1: {metrics['macro_f1']:.4f}")

    model_path = Path(cfg["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"model": pipe, "meta": {"config": cfg, "metrics": metrics, "id2label": ID2LABEL}}
    joblib.dump(bundle, model_path)
    print(f"Сохранена модель: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение baseline: TF-IDF + LogisticRegression.")
    parser.add_argument("--config", default="configs/train_baseline.json", help="JSON-конфиг.")
    args = parser.parse_args()
    main(args.config)
