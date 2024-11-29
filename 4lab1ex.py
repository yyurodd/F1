'''
Лабораторная работа 4 - Задача регрессии на данных Формулы-1

Этот скрипт решает задачу регрессии для предсказания финишной позиции гонщика
на основе различных параметров гонки.

Основные шаги:
1. Загрузка и подготовка данных
2. Разделение на входные параметры (X) и целевую метку (y)
3. Разделение на обучающую и тестовую выборки
4. Обучение модели линейной регрессии
5. Оценка качества модели
6. Визуализация результатов
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.simplefilter('ignore')
sns.set()

# Загрузка данных
drivers = pd.read_csv("AI/F1/drivers.csv")
results = pd.read_csv("AI/F1/results.csv")
qualifying = pd.read_csv("AI/F1/qualifying.csv")
races = pd.read_csv("AI/F1/races.csv")
circuits = pd.read_csv("AI/F1/circuits.csv")

# Объединение данных
df = pd.merge(drivers, results, on='driverId')
df = pd.merge(df, races, on='raceId')
df = pd.merge(df, circuits, on='circuitId')
df['dob'] = df['dob'].str[:4]

# Подготовка данных: выбор признаков для предсказания
features = ['grid', 'laps', 'dob']  # входные параметры
target = 'positionOrder'     # целевая метка

# Удаление строк с пропущенными значениями
df = df.dropna(subset=features + [target])

# Разделение на входные параметры (X) и целевую метку (y)
X = df[features]
y = df[target]

# Разделение на обучающую и тестовую выборки (80% на обучение, 20% на тест)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели линейной регрессии
model = LinearRegression()
#positionOrder = a*grid + b*laps + c
model.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred = model.predict(X_test)

# Оценка качества модели
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Результаты оценки модели:")
print(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")
print(f"Коэффициент детерминации (R²): {r2:.2f}")

# Визуализация результатов
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Реальная позиция')
plt.ylabel('Предсказанная позиция')
plt.title('Сравнение реальных и предсказанных позиций')
plt.tight_layout()
plt.show()

# Визуализация важности признаков
importance = pd.DataFrame({
    'Признак': features,
    'Важность': model.coef_
})
plt.figure(figsize=(8, 4))
sns.barplot(data=importance, x='Важность', y='Признак')
plt.title('Важность признаков в модели')
plt.tight_layout()
plt.show()

# Вывод коэффициентов модели
print("\nКоэффициенты модели:")
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"Свободный член: {model.intercept_:.4f}")