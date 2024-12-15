'''
Лабораторная работа 4 - Задача классификации на данных Формулы-1

Задача: предсказать, попадет ли гонщик на подиум (топ-3) на основе различных характеристик.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from sklearn.preprocessing import LabelEncoder
import warnings

warnings.simplefilter('ignore')
sns.set()

# Загрузка данных
drivers = pd.read_csv("AI/F1/drivers.csv")
results = pd.read_csv("AI/F1/results.csv")
qualifying = pd.read_csv("AI/F1/qualifying.csv")
races = pd.read_csv("AI/F1/races.csv")
circuits = pd.read_csv("AI/F1/circuits.csv")
constructors = pd.read_csv("AI/F1/constructors.csv")

# Объединение данных
df = pd.merge(drivers, results, on='driverId')
df = pd.merge(df, races, on='raceId', suffixes=('', '_race'))
df = pd.merge(df, circuits, on='circuitId', suffixes=('', '_circuit'))
df = pd.merge(df, constructors, on='constructorId', suffixes=('', '_constructor'))


# Подготовка данных
df['dob'] = pd.to_datetime(df['dob']).dt.year
df['date'] = pd.to_datetime(df['date']).dt.year
df['age'] = df['date'] - df['dob']
df['experience'] = df.groupby('driverId')['raceId'].transform('rank')
df['podium'] = (df['positionOrder'] <= 3).astype(int)

# Кодирование категориальных признаков
le = LabelEncoder()
df['constructor_encoded'] = le.fit_transform(df['constructorRef'])
df['circuit_encoded'] = le.fit_transform(df['circuitRef'])

# Подготовка данных: выбор признаков для предсказания
features = [
    'grid',                 # стартовая позиция
    'laps',                # количество кругов
    'age',                 # возраст гонщика
    'experience',          # опыт (количество гонок)
    'constructor_encoded', # команда
    'circuit_encoded'      # трасса
]
target = 'podium'

# Удаление строк с пропущенными значениями
df = df.dropna(subset=features + [target])

# Разделение на входные параметры (X) и целевую метку (y)
X = df[features]
y = df[target]

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели Random Forest с учетом дисбаланса классов
model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred = model.predict(X_test)

# Оценка качества модели
accuracy = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)


print("Результаты оценки модели:")
print(f"Точность: {accuracy:.3f}")
print("\nМатрица ошибок:")
print(conf_matrix)


# Визуализация матрицы ошибок
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Матрица ошибок')
plt.xlabel('Предсказанные значения')
plt.ylabel('Реальные значения')
plt.show()

# Визуализация важности признаков
importance = pd.DataFrame({
    'Признак': features,
    'Важность': model.feature_importances_
})
importance = importance.sort_values('Важность', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance, x='Важность', y='Признак')
plt.title('Важность признаков в модели')
plt.tight_layout()
plt.show()

# Дополнительная статистика
podium_ratio = df['podium'].mean()
print(f"\nСтатистика данных:")
print(f"Количество записей (результатов гонщиков): {len(df)}")
print(f"Количество уникальных гонок: {df['raceId'].nunique()}")
print(f"Количество подиумов: {df['podium'].sum()}")
print(f"Среднее количество гонщиков в гонке: {len(df) / df['raceId'].nunique():.1f}")
