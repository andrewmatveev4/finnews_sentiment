from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from finnews_sentiment.models.train_model import load_config, split_data
from finnews_sentiment.features.preprocess import ID2LABEL

BEST_MODEL_PATH = Path("models/best_model.joblib")
FIGURES_DIR = Path("reports/figures")
LABELS = [ID2LABEL[i] for i in sorted(ID2LABEL)]
PATTERNS = {
    "сравнение (compared to/with)": r"compared (?:to|with)",
    "слово loss": r"\bloss",
    "слово cut": r"\bcut\b",
    "слово negative": r"\bnegative\b",
}


def build_predictions(model, id2label: dict, X_test: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    """Таблица по test: текст, правильная метка, ответ модели, уверенность, верно ли."""
    proba = model.predict_proba(X_test)
    pred = model.classes_[proba.argmax(axis=1)]
    return pd.DataFrame({
        "text": X_test.values,
        "true": [id2label[i] for i in y_test],
        "pred": [id2label[i] for i in pred],
        "confidence": proba.max(axis=1).round(3),
        "correct": y_test.values == pred,
    })


def confusion_tables(preds: pd.DataFrame) -> tuple:
    """Confusion matrix в штуках и в долях по строке (диагональ долей = recall)."""
    counts = pd.crosstab(preds["true"], preds["pred"]).reindex(
        index=LABELS, columns=LABELS, fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0).round(3)
    return counts, shares


def plot_confusion(counts: pd.DataFrame, shares: pd.DataFrame) -> Path:
    """Тепловая карта: цвет по долям, подписи 'штуки и %'."""
    annot = counts.astype(str) + "\n" + (shares * 100).round().astype(int).astype(str) + "%"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "confusion_matrix.png"
    plt.figure(figsize=(6, 5))
    sns.heatmap(shares, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1)
    plt.xlabel("Ответ модели")
    plt.ylabel("Правильная метка")
    plt.title("Confusion matrix (цвет = доля строки)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    return out_path


def top_words(model, n: int = 15) -> pd.DataFrame:
    """Для каждого класса n слов и пар слов с самыми большими весами логрега."""
    vocab = model.named_steps["tfidf"].get_feature_names_out()
    clf = model.named_steps["clf"]
    columns = {}
    for row, cls in enumerate(clf.classes_):
        best = clf.coef_[row].argsort()[::-1][:n]
        columns[ID2LABEL[cls]] = vocab[best]
    return pd.DataFrame(columns)


def confident_errors(preds: pd.DataFrame, true_label: str, pred_label: str,
                     n: int = 5) -> pd.DataFrame:
    """n ошибок 'правда true_label -> ответ pred_label' с самой высокой уверенностью."""
    mask = (preds["true"] == true_label) & (preds["pred"] == pred_label)
    return preds[mask].sort_values("confidence", ascending=False).head(n)


def error_rate_by_pattern(preds: pd.DataFrame, pattern: str) -> pd.DataFrame:
    """Доля ошибок в текстах с паттерном и без него."""
    has = preds["text"].str.contains(pattern, case=False, regex=True)
    rows = {
        "с паттерном": preds.loc[has, "correct"],
        "без паттерна": preds.loc[~has, "correct"],
    }
    return pd.DataFrame({
        "текстов": {k: len(v) for k, v in rows.items()},
        "доля ошибок": {k: round(1 - v.mean(), 3) for k, v in rows.items()},
    })


if __name__ == "__main__":
    cfg = load_config(Path("configs/train_baseline.json"))
    _, X_test, _, y_test = split_data(pd.read_csv(cfg["input_path"]), cfg)
    bundle = joblib.load(BEST_MODEL_PATH)
    preds = build_predictions(bundle["model"], bundle["meta"]["id2label"], X_test, y_test)
    print(preds.head().to_string())
    print(f"\nОшибок: {(~preds['correct']).sum()} из {len(preds)}")
    counts, shares = confusion_tables(preds)
    print("\nШтуки:\n", counts)
    print("\nДоли по строке:\n", shares)
    print(f"\nГрафик: {plot_confusion(counts, shares)}")
    print("\nТоп слов по классам:\n", top_words(bundle["model"]).to_string())
    for true_label, pred_label in [("negative", "positive"), ("positive", "negative"),
                                   ("neutral", "positive")]:
        print(f"\n=== Правда {true_label} -> модель сказала {pred_label} ===")
        for _, row in confident_errors(preds, true_label, pred_label).iterrows():
            print(f"[{row['confidence']:.2f}] {row['text']}")
    for name, pattern in PATTERNS.items():
        print(f"\n{name}:")
        print(error_rate_by_pattern(preds, pattern))
