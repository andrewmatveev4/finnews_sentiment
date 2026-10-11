# День 3. Baseline: TF-IDF + Logistic Regression

## Постановка
- Данные: data/processed/clean_news.csv (4835 строк).
- Разбиение: train 3868 / test 967, stratify, random_state=42.
  В test: negative 121, neutral 574, positive 272.
- Основная метрика: macro F1.

## Модель
Pipeline: preprocess_text (внутри TfidfVectorizer) -> TF-IDF (max_features=5000, ngram_range=(1, 2))
-> LogisticRegression(max_iter=1000). Обучение на сыром тексте, чистка встроена в модель.

## Результаты

| Модель | Accuracy | Macro F1 |
|---|---|---|
| Dummy (всегда neutral) | 0.594 | 0.248 |
| **Baseline TF-IDF + LogReg** | **0.776** | **0.6925** |

| Класс | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| negative | 0.841 | 0.438 | 0.576 | 121 |
| neutral | 0.768 | 0.949 | 0.849 | 574 |
| positive | 0.784 | 0.559 | 0.652 | 272 |

## Наблюдения
- Модель смещена к neutral: предсказала neutral 710 раз при 574 реальных.
- Negative: точно (precision 0.84), но находит меньше половины (recall 0.44).
- TF-IDF по умолчанию отбрасывает `%` и однозначные числа (токен = 2+ буквенно-цифровых символа).
- Незнакомые слова (не из словаря train) игнорируются без ошибки.

## Цель Дня 4
Улучшить macro F1 на 5%: относительно -> 0.727, в процентных пунктах -> 0.7425.

## Артефакты
- models/baseline_logreg.joblib (модель + конфиг, метрики, id2label; не в git, пересобирается командой ниже).
- Команда: `python -m finnews_sentiment.models.train_model`