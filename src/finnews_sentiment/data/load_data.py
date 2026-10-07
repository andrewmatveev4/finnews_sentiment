import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def load_phrasebank(path: Path) -> pd.DataFrame:
    """Читает Financial PhraseBank: строки вида 'текст@метка', кодировка latin-1."""
    rows = []
    with open(path, encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text, sentiment = line.rsplit("@", 1)
            rows.append({"text": text, "sentiment": sentiment})
    return pd.DataFrame(rows)


def save_processed(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main(input_path: str, output_path: str, sample: Optional[int] = None) -> None:
    input_p = Path(input_path)
    output_p = Path(output_path)

    df = load_phrasebank(input_p)
    if sample is not None and sample > 0:
        df = df.sample(n=sample, random_state=42)
    save_processed(df, output_p)
    print(f"Сохранён распарсенный датасет: {output_p} (shape={df.shape})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Базовая загрузка и очистка новостей.")
    parser.add_argument("--input", type=str, required=True,
                        help="Путь к исходному txt-файлу Financial PhraseBank.")
    parser.add_argument("--output", type=str, required=True,
                        help="Путь для сохранения распарсенного CSV.")
    parser.add_argument("--sample", type=int, default=None,
                        help="Опционально: взять случайный сэмпл N строк.")
    args = parser.parse_args()

    main(args.input, args.output, sample=args.sample)
