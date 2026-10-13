# День 5. Error analysis

Модель: E6 (LogReg, C=3, class_weight="balanced"), тот же test (967), что на Днях 3-4.
Test для подстройки больше не используется.

## Confusion matrix
![confusion matrix](figures/confusion_matrix.png)

| правда \ ответ | negative | neutral | positive |
|---|---|---|---|
| negative (121) | **83** (69%) | 25 (21%) | 13 (11%) |
| neutral (574) | 23 (4%) | **476** (83%) | 75 (13%) |
| positive (272) | 22 (8%) | 65 (24%) | **185** (68%) |

Ошибок 223 из 967.
- neutral <-> positive: 140 ошибок (63%) - главный источник.
- neutral <-> negative: 48 (22%).
- negative <-> positive: 35 (16%) - редкие, но самые дорогие для трейдера.
- Модель называет neutral 566 раз при 574 реальных (baseline: 710), смещение к neutral устранено.

## Что выучила модель (топ весов логрега)
- negative: down, decreased, fell, lower, dropped, lay, off, union, staff
- positive: increased, rose, grew, growth, improved, signed, awarded, up, up from
- neutral: is, approximately, includes, range, will, no, not

`down` - самый сильный признак negative, `up` / `up from` - в топе positive. Это слова, которые
удалял шаблонный препроцессинг (стоп-слова NLTK); решение Дня 2 подтверждено весами модели.
В топе neutral есть название компании `tikkurila` - модель запомнила компанию, а не тональность.

## Причины ошибок (по 15 самым уверенным ошибкам)
1. **Направление без объекта и сравнения.** «increased fixed costs» -> positive; «loss of 0.4 mn compared
   to a loss of 1.9 mn» -> negative; «cash flow 0.4 mn compared to a negative 15.5 mn» -> negative.
   Мешок слов не видит, что выросло, и не сравнивает числа.
2. **Многозначные слова.** `up` в «coming up with»; `cut` для расходов и для прогнозов;
   `negative` как знак числа.
3. **Синонимы и образный язык.** «Export declined», «uncertainties cloud the outlook».
4. **Спорная разметка (50Agree).** «pleased to welcome the new employees» и «increase copper capacity»
   размечены neutral.

## Гипотезы Дня 1
| Гипотеза | Статус |
|---|---|
| негативные слова -> позитивный итог | подтверждена примерами (отмена ограничений, сокращение убытка) |
| прибыль без направления -> neutral | в топе ошибок не встретилась, не проверена |
| спорная разметка экспертов | подтверждена примерами |

## Проверка паттернов на всём test
| Паттерн | Текстов | Доля ошибок | Без паттерна |
|---|---|---|---|
| compared to/with | 27 | 0.296 | 0.229 |
| loss | 27 | 0.296 | 0.229 |
| cut | 7 | 0.429 | 0.229 |
| negative | 4 | 0.500 | 0.229 |

Все группы ошибаются чаще среднего, но выборки 4-27 текстов: при n=27 шум около +-8 п.п.
Направление подтверждает чтение примеров, статистически не доказано.
Группы «compared» и «loss» пересекаются только на 9 текстах (совпадение 27/27 случайно).

## Что улучшать дальше
- Модель с учётом контекста (трансформер), которая видит «loss X compared to loss Y».
- Больше примеров для синонимов направления (declined, slumped, soared).
- Маскировать названия компаний, чтобы модель не запоминала `tikkurila`.
- Перепроверить разметку уверенных ошибок: часть из них - шум данных, а не ошибки модели.

## Артефакты
- reports/figures/confusion_matrix.png
- reports/day05_error_analysis_output.txt (полный вывод скрипта)
- Команда: `python -m finnews_sentiment.models.error_analysis`