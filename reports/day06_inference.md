# День 6. Inference pipeline

## Модель
`models/best_model.joblib`: один файл с Pipeline (чистка текста -> TF-IDF -> LogReg, C=3, balanced)
и паспортом (эксперимент, метрики test, таблица CV, id2label). Vectorizer внутри Pipeline,
поэтому модель и векторайзер нельзя рассинхронизировать.
Требование: пакет установлен (`pip install -e .`), так как модель хранит ссылку на `preprocess_text`.

## Как пользоваться

Одна новость:
    python -m finnews_sentiment.models.predict --text "Net profit fell 30 % in the third quarter ."

Файл:
    python -m finnews_sentiment.models.predict --input examples/new_news.csv --output reports/predictions.csv

Из Python:
    from finnews_sentiment.models.predict import load_model, predict
    model, meta = load_model()
    predict("Net profit fell 30 % .", model, meta["id2label"])

## Формат входа
- Строка или список строк на английском. Пустая строка, пробелы, не-строка -> ValueError.
- CSV: обязательна колонка `text`, остальные колонки сохраняются в выходном файле.

## Формат выхода
Всегда список словарей, по одному на новость, в том же порядке:

    {
      "text": "Net profit FELL 30 % in the third quarter .",
      "label": "negative",
      "confidence": 0.945,
      "probabilities": {"negative": 0.945, "neutral": 0.029, "positive": 0.026},
      "unknown_words": []
    }

- `label`: negative / neutral / positive.
- `confidence`: максимальная вероятность.
- `probabilities`: все три вероятности, сумма 1.
- `unknown_words`: слова новости, которых нет в словаре модели (вес 0).

CSV: исходные колонки + `pred_label`, `confidence`, `proba_negative`, `proba_neutral`, `proba_positive`,
`unknown_words` (через запятую). Пустые `unknown_words` pandas читает как NaN:
`pd.read_csv(path, keep_default_na=False)`.

## Как читать ответ
- Низкий `confidence` (ниже ~0.7): знакомые слова тянут в разные стороны (сравнения, «рост расходов»).
- Непустой `unknown_words`: модель не видит часть слов и может уверенно ошибиться.
- Вероятности сдвинуты class_weight и не откалиброваны: 0.9 не означает «верно в 90% случаев».

## Демо: examples/new_news.csv (8 новостей, 5 из них ловушки из Дня 5)
| Новость | Ответ | confidence | unknown_words | Верно |
|---|---|---|---|---|
| Net profit fell 30 % | negative | 0.945 | - | да |
| annual general meeting on Tuesday | neutral | 0.688 | - | да |
| Sales rose sharply, strong demand | positive | 0.973 | sharply | да |
| loss 0.4 compared to loss 1.9 | negative | 0.558 | - | нет |
| Russia will cancel import restrictions | neutral | 0.436 | cancel, import, restrictions, dairy | нет |
| Operating costs increased by 15 % | positive | 0.563 | - | нет |
| Shares plummeted after the CEO resigned | neutral | 0.872 | plummeted, resigned | нет |
| no plans to cut jobs | negative | 0.683 | - | нет |

Все 5 ошибок помечены хотя бы одним сигналом: низкий confidence (4 случая) или незнакомые слова (2).

## Тесты
26 тестов (10 на inference). Тесты inference используют маленькую модель из фикстуры
и не зависят от файла `models/`, поэтому проходят сразу после клонирования.