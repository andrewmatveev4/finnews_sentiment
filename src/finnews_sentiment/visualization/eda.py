from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

INTERIM_PATH = Path("data/interim/phrasebank_50agree.csv")
FIGURES_DIR = Path("reports/figures")


def describe_dataset(df: pd.DataFrame) -> None:
    """Печатает размер, типы колонок и количество пропусков."""
    print(f"Размер: {df.shape[0]} строк, {df.shape[1]} колонок")
    print("\nТипы колонок:")
    print(df.dtypes)
    print("\nПропуски (NaN) по колонкам:")
    print(df.isna().sum())


def check_empty_texts(df: pd.DataFrame, text_col: str = "text") -> None:
    """Считает пустые и очень короткие тексты, которые isna() не видит."""
    stripped = df[text_col].str.strip()
    word_count = stripped.str.split().str.len()
    print(f"\nПустых текстов (после strip): {(stripped == '').sum()}")
    print(f"Текстов короче 3 слов: {(word_count < 3).sum()}")
    print(df.loc[word_count < 3, text_col].head(10).to_string())


def check_duplicates(df: pd.DataFrame, text_col: str = "text",
                     target_col: str = "sentiment") -> None:
    """Считает полные дубли и дубли текста с разными метками."""
    full_dups = df.duplicated(subset=[text_col, target_col]).sum()
    text_dups = df.duplicated(subset=[text_col]).sum()
    print(f"\nПолных дублей (текст + метка): {full_dups}")
    print(f"Дублей текста с другой меткой: {text_dups - full_dups}")
    n_labels = df.groupby(text_col)[target_col].transform("nunique")
    print(df[n_labels > 1].sort_values(text_col).to_string())


def plot_class_distribution(df: pd.DataFrame, target_col: str = "sentiment") -> None:
    """Печатает число и долю каждого класса и сохраняет график."""
    counts = df[target_col].value_counts()
    shares = df[target_col].value_counts(normalize=True).round(3)
    print("\nРаспределение классов:")
    print(pd.DataFrame({"count": counts, "share": shares}))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "class_distribution.png"
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=target_col, order=["negative", "neutral", "positive"])
    plt.title("Распределение классов")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"График сохранён: {out_path}")


def analyze_text_lengths(df: pd.DataFrame, text_col: str = "text",
                         target_col: str = "sentiment") -> None:
    """Считает длины текстов в символах и словах, печатает статистику, сохраняет гистограмму."""
    lengths = pd.DataFrame({
        "chars": df[text_col].str.len(),
        "words": df[text_col].str.split().str.len(),
        target_col: df[target_col],
    })
    print("\nДлины текстов (все классы):")
    print(lengths[["chars", "words"]].describe(percentiles=[0.5, 0.95, 0.99]).round(1))
    print("\nСлов в среднем по классам:")
    print(lengths.groupby(target_col)["words"].agg(["mean", "median", "max"]).round(1))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "text_length_distribution.png"
    plt.figure(figsize=(8, 4))
    sns.histplot(data=lengths, x="words", hue=target_col, bins=40, element="step")
    plt.title("Распределение длины текстов (в словах)")
    plt.xlabel("Количество слов (знаки препинания тоже считаются)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"График сохранён: {out_path}")


def show_examples(df: pd.DataFrame, text_col: str = "text", target_col: str = "sentiment",
                  n: int = 3, seed: int = 42) -> None:
    """Печатает n случайных примеров каждого класса и самый длинный текст."""
    for label in ["negative", "neutral", "positive"]:
        print(f"\n=== {label} ===")
        sample = df[df[target_col] == label].sample(n=n, random_state=seed)
        for text in sample[text_col]:
            print(f"- {text}")

    longest = df.loc[df[text_col].str.split().str.len().idxmax()]
    print(f"\n=== Самый длинный текст ({longest[target_col]}) ===\n{longest[text_col]}")


if __name__ == "__main__":
    df = pd.read_csv(INTERIM_PATH)
    describe_dataset(df)
    check_empty_texts(df)
    check_duplicates(df)
    plot_class_distribution(df)
    analyze_text_lengths(df)
    show_examples(df)
