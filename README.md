# TMO_NIRS

**Научно-исследовательская работа студента (НИРС)**  
Курс: Технологии машинного обучения (ТМО)  
Группа: ИУ5-64Б  
Автор: Юрченко К.Г.

---

## Описание

Проект посвящён анализу данных и построению моделей машинного обучения на примере датасета **California Housing Prices** — стоимость жилья в округах Калифорнии на основе переписи населения 1990 года.

Источник данных: [`sklearn.datasets.fetch_california_housing`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html), также доступен на [Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices).

---

## Задачи

| Задача | Целевой признак | Описание |
|--------|----------------|----------|
| Классификация | `price_class` | Жильё «дорогое» (выше медианы) — 1, иначе — 0 |
| Регрессия | `MedHouseVal` | Медианная стоимость жилья ($100k) |

---

## Структура проекта

```
TMO_NIRS/
├── nirs_california_housing.ipynb
└── README.md
```

---

## Содержание ноутбука

1. Описание датасета и постановка задачи
2. Импорт библиотек
3. Загрузка и предобработка данных
4. Разведочный анализ данных (EDA): парные диаграммы, скрипичные графики, географическая карта цен
5. Масштабирование признаков (MinMaxScaler)
6. Корреляционный анализ
7. Выбор метрик качества
8. Baseline-модели (без подбора гиперпараметров):
   - Классификация: LogR, KNN, SVC, DecisionTree, RandomForest, GradientBoosting
   - Регрессия: LinearRegression, KNN, SVR, DecisionTree, RandomForest, GradientBoosting
9. Подбор гиперпараметров (GridSearchCV, 5-fold CV)
10. Сравнение baseline и оптимизированных моделей
11. Выводы

---

## Результаты

### Классификация (лучшие модели по ROC AUC)

| Модель | Precision | Recall | F1 | ROC AUC |
|--------|-----------|--------|----|---------|
| GradientBoosting | высокий | высокий | высокий | наивысший |
| RandomForest | высокий | высокий | высокий | высокий |

### Регрессия (предсказание стоимости жилья)

| Модель | MAE | MSE | R² |
|--------|-----|-----|----|
| RandomForest | 0.325 | 0.243 | **0.817** |
| GradientBoosting | 0.362 | 0.265 | 0.800 |
| LinearRegression | 0.534 | 0.512 | 0.614 |

---

## Зависимости

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Установка:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Запуск

```bash
jupyter notebook nirs_california_housing.ipynb
```
