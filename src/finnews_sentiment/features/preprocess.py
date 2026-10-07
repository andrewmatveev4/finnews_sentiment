import re
import argparse
from pathlib import Path

import pandas as pd

HTML_TAG = re.compile(r"<[^>]+>")
SPLIT_NUMBER = re.compile(r"(\d)\s+\.(\d)")
WHITESPACE = re.compile(r"\s+")
# грубое приближение тикеров, только для сырого текста (после lower() не работает)
TICKER_PATTERN = re.compile(r"\b[A-Z]{2,6}\b")
MIN_WORDS = 3
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {i: name for name, i in LABEL2ID.items()}


def preprocess_text(text: str) -> str:
    """Базовая чистка без потери тональности.

    Шаги: нижний регистр, HTML-теги в пробел, склейка разорванных чисел ('5 .9' -> '5.9'),
    схлопывание пробелов. Стоп-слова, цифры и пунктуация сохраняются: в них сигнал тональности.
    """
    if not isinstance(text, str):
        return ""
    txt = text.lower()
    txt = HTML_TAG.sub(" ", txt)
    txt = SPLIT_NUMBER.sub(r"\1.\2", txt)
    txt = WHITESPACE.sub(" ", txt)
    return txt.strip()


def clean_news_data(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Применяет preprocess_text ко всему DataFrame.

    Создаёт новую колонку `text_clean`.
    """
    df = df.copy()
    df["text_clean"] = df[text_col].astype(str).map(preprocess_text)
    return df


def add_basic_features(df: pd.DataFrame, text_col: str = "text_clean") -> pd.DataFrame:
    """Добавляет простые числовые признаки на основе текста.

    Примеры:
    - длина текста в словах
    - количество тикеров
    - количество чисел в исходном тексте
    """
    df = df.copy()
    df["text_len_words"] = df[text_col].astype(str).str.split().apply(len)
    df["ticker_count"] = df["text"].astype(str).apply(lambda s: len(TICKER_PATTERN.findall(s)))

    # количество чисел в исходном тексте
    df["number_count"] = df["text"].astype(str).str.findall(r"\d+(?:\.\d+)?").apply(len)

    return df


def drop_bad_rows(df: pd.DataFrame, text_col: str = "text_clean",
                  target_col: str = "sentiment") -> pd.DataFrame:
    """Удаляет короткие тексты, полные дубли и тексты с конфликтующими метками."""
    n_start = len(df)
    df = df[df[text_col].str.split().str.len() >= MIN_WORDS]
    n_short = n_start - len(df)

    df = df.drop_duplicates(subset=[text_col, target_col])
    n_dups = n_start - n_short - len(df)

    n_labels = df.groupby(text_col)[target_col].transform("nunique")
    df = df[n_labels == 1]
    n_conflicts = n_start - n_short - n_dups - len(df)

    print(f"Удалено: коротких {n_short}, дублей {n_dups}, конфликтных {n_conflicts}. "
          f"Осталось {len(df)} из {n_start}")
    return df.reset_index(drop=True)


def encode_labels(df: pd.DataFrame, target_col: str = "sentiment") -> pd.DataFrame:
    """Добавляет колонку label с числовым кодом класса. Падает на неизвестной метке."""
    df = df.copy()
    df["label"] = df[target_col].map(LABEL2ID)
    unknown = df.loc[df["label"].isna(), target_col].unique()
    if len(unknown) > 0:
        raise ValueError(f"Неизвестные метки: {list(unknown)}")
    df["label"] = df["label"].astype(int)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Чистка текстов и строк датасета.")
    parser.add_argument("--input", required=True, help="Распарсенный CSV из data/interim.")
    parser.add_argument("--output", required=True, help="Куда сохранить очищенный CSV.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = clean_news_data(df, text_col="text")
    df = drop_bad_rows(df)
    df = encode_labels(df)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Сохранён очищенный датасет: {output} (shape={df.shape})")
