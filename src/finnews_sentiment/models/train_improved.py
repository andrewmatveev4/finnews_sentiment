from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from finnews_sentiment.features.preprocess import ID2LABEL, preprocess_text
import pandas as pd
import argparse
from pathlib import Path
from finnews_sentiment.models.train_model import evaluate, load_config, split_data
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
import joblib

RESULTS_PATH = Path("reports/day04_cv_results.csv")
BEST_MODEL_PATH = Path("models/best_model.joblib")


def make_pipeline(clf=None, **tfidf_params) -> Pipeline:
    """Baseline-пайплайн, в котором можно заменить модель и любые параметры TF-IDF."""
    params = {"preprocessor": preprocess_text, "max_features": 5000, "ngram_range": (1, 2)}
    params.update(tfidf_params)
    if clf is None:
        clf = LogisticRegression(max_iter=1000)
    return Pipeline([("tfidf", TfidfVectorizer(**params)), ("clf", clf)])


def cv_score(pipe: Pipeline, X_train: pd.Series, y_train: pd.Series, seed: int) -> tuple:
    """5-fold стратифицированная кросс-валидация по macro F1 только на train."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
    return scores.mean(), scores.std()


def tune_logreg(X_train: pd.Series, y_train: pd.Series, seed: int) -> GridSearchCV:
    """Подбор C и class_weight логрега по macro F1 на кросс-валидации train."""
    grid = GridSearchCV(
        make_pipeline(),
        param_grid={"clf__C": [0.3, 1, 3, 10, 30], "clf__class_weight": [None, "balanced"]},
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
    )
    return grid.fit(X_train, y_train)


def get_experiments() -> dict:
    """Имя эксперимента -> Pipeline. Каждый меняет одну вещь относительно baseline."""
    return {
        "baseline": make_pipeline(),
        "E1_class_weight": make_pipeline(
            clf=LogisticRegression(max_iter=1000, class_weight="balanced")),
        "E2_token_pattern": make_pipeline(token_pattern=r"(?u)\b\w+\b|%"),
        "E3_trigrams": make_pipeline(ngram_range=(1, 3)),
        "E4_linear_svc": make_pipeline(clf=LinearSVC(random_state=42)),
        "E5_tuned_C10": make_pipeline(clf=LogisticRegression(max_iter=1000, C=10)),
        "E6_C3_balanced": make_pipeline(
            clf=LogisticRegression(max_iter=1000, C=3, class_weight="balanced")),
    }


def main(config_path: str) -> None:
    cfg = load_config(Path(config_path))
    df = pd.read_csv(cfg["input_path"])
    X_train, X_test, y_train, y_test = split_data(df, cfg)

    rows = []
    for name, pipe in get_experiments().items():
        mean, std = cv_score(pipe, X_train, y_train, cfg["random_state"])
        rows.append({"experiment": name, "cv_macro_f1": round(mean, 4), "cv_std": round(std, 4)})
        print(f"{name:<20} {mean:.4f} +- {std:.4f}")

    results = pd.DataFrame(rows).sort_values("cv_macro_f1", ascending=False)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    print(f"Результаты: {RESULTS_PATH}")

    best_name = results.iloc[0]["experiment"]
    best_pipe = get_experiments()[best_name]
    best_pipe.fit(X_train, y_train)
    print(f"\nЛучший по CV: {best_name}. Единственная проверка на test:")
    test_metrics = evaluate(best_pipe, X_test, y_test)

    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    meta = {"experiment": best_name, "config": cfg, "test_metrics": test_metrics,
            "cv_results": results.to_dict("records"), "id2label": ID2LABEL}
    joblib.dump({"model": best_pipe, "meta": meta}, BEST_MODEL_PATH)
    print(f"Сохранена модель: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сравнение улучшений baseline по кросс-валидации.")
    parser.add_argument("--config", default="configs/train_baseline.json", help="JSON-конфиг.")
    args = parser.parse_args()
    main(args.config)
