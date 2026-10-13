import argparse
import json
from pathlib import Path
import joblib
import pandas as pd

from finnews_sentiment.models.train_improved import BEST_MODEL_PATH


def load_model(path: Path = BEST_MODEL_PATH) -> tuple:
    """Загружает модель и паспорт. Понятная ошибка, если модель ещё не обучена."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Модель не найдена: {path}. "
            "Сначала обучите: python -m finnews_sentiment.models.train_improved")
    bundle = joblib.load(path)
    return bundle["model"], bundle["meta"]


def unknown_words(text: str, model) -> list:
    """Слова новости, которых нет в словаре модели: их вес нулевой, модель их не видит."""
    tfidf = model.named_steps["tfidf"]
    tokens = tfidf.build_analyzer()(text)
    words = [t for t in tokens if " " not in t and t not in tfidf.vocabulary_]
    return list(dict.fromkeys(words))


def predict(texts, model, id2label: dict) -> list:
    """Классифицирует новость (str) или список новостей. Всегда возвращает список словарей."""
    if isinstance(texts, str):
        texts = [texts]
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Ожидается непустой текст, получено: {text!r}")

    proba = model.predict_proba(texts)
    results = []
    for text, row in zip(texts, proba):
        best = row.argmax()
        results.append({
            "text": text,
            "label": id2label[model.classes_[best]],
            "confidence": round(float(row[best]), 3),
            "probabilities": {id2label[c]: round(float(p), 3)
                              for c, p in zip(model.classes_, row)},
            "unknown_words": unknown_words(text, model),
        })
    return results


def predict_file(input_path: str, output_path: str, model, id2label: dict,
                 text_col: str = "text") -> pd.DataFrame:
    """CSV с колонкой text -> тот же CSV с колонками ответа модели."""
    df = pd.read_csv(input_path)
    if text_col not in df.columns:
        raise ValueError(f"В {input_path} нет колонки '{text_col}', есть: {list(df.columns)}")

    results = predict(df[text_col].tolist(), model, id2label)
    out = df.copy()
    out["pred_label"] = [r["label"] for r in results]
    out["confidence"] = [r["confidence"] for r in results]
    for name in id2label.values():
        out[f"proba_{name}"] = [r["probabilities"][name] for r in results]
        out["unknown_words"] = [", ".join(r["unknown_words"]) for r in results]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Тональность финансовых новостей.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--text", help="Одна новость на английском.")
    mode.add_argument("--input", help="CSV с колонкой text.")
    parser.add_argument("--output", default="reports/predictions.csv", help="Куда сохранить CSV.")
    parser.add_argument("--model", default=str(BEST_MODEL_PATH), help="Путь к модели.")
    args = parser.parse_args()

    model, meta = load_model(args.model)
    if args.text:
        print(json.dumps(predict(args.text, model, meta["id2label"]), indent=2, ensure_ascii=False))
    else:
        out = predict_file(args.input, args.output, model, meta["id2label"])
        print(out[["pred_label", "confidence", "text"]].to_string(index=False))
        print(f"\nСохранено: {args.output}")
