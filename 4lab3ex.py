'''
Лабораторная работа 4 - Задача кластеризации на данных Формулы-1

Задача: Выполнить кластеризацию гонщиков на основе их характеристик и результатов,
исключая информацию о подиумах.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

warnings.simplefilter('ignore')
sns.set()

# Загрузка данных
drivers = pd.read_csv("AI/F1/drivers.csv")
results = pd.read_csv("AI/F1/results.csv")
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

# Правильный расчет опыта: сортируем по году и номеру этапа в сезоне
df = df.sort_values(['driverId', 'year', 'round'])
df['experience'] = df.groupby('driverId').cumcount()

# Преобразование и очистка числовых столбцов
df['grid'] = pd.to_numeric(df['grid'], errors='coerce')
df['position'] = pd.to_numeric(df['position'], errors='coerce')
df['points'] = pd.to_numeric(df['points'], errors='coerce')
df['laps'] = pd.to_numeric(df['laps'], errors='coerce')

# Удаление строк с пропущенными значениями
df = df.dropna(subset=['grid', 'position', 'points', 'laps', 'age'])

# Агрегация данных по гонщикам (берем средние значения)
driver_stats = df.groupby('driverId').agg({
    'age': 'mean',
    'grid': 'mean',
    'position': 'mean',
    'points': 'mean',
    'experience': 'max',  # берем максимальное значение опыта
    'laps': 'mean'
}).reset_index()

# Выбор признаков для кластеризации
features = ['grid', 'position', 'points', 'age', 'experience', 'laps']
X = driver_stats[features]

# Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Определение оптимального количества кластеров методом локтя
inertias = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# График метода локтя
plt.figure(figsize=(10, 6))
plt.plot(K, inertias, 'bx-')
plt.xlabel('k')
plt.ylabel('Инерция')
plt.title('Метод локтя для определения оптимального k')
plt.show()

# Применение KMeans с оптимальным количеством кластеров
n_clusters = 4  # выбираем на основе графика
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
driver_stats['Cluster'] = kmeans.fit_predict(X_scaled)

# Добавим имена гонщиков для анализа
driver_names = df[['driverId', 'forename', 'surname']].drop_duplicates()
driver_stats = pd.merge(driver_stats, driver_names, on='driverId')
driver_stats['full_name'] = driver_stats['forename'] + ' ' + driver_stats['surname']

# Визуализация результатов
# График 1: Средняя позиция vs Средняя стартовая позиция
plt.figure(figsize=(12, 8))
scatter = plt.scatter(driver_stats['grid'], 
                     driver_stats['position'],
                     c=driver_stats['Cluster'],
                     cmap='viridis',
                     s=100)
plt.xlabel('Средняя стартовая позиция')
plt.ylabel('Средняя финишная позиция')
plt.title('Кластеры гонщиков: Стартовая vs Финишная позиция')
plt.colorbar(scatter, label='Кластер')
plt.show()

# График 2: Опыт vs Очки
plt.figure(figsize=(12, 8))
scatter = plt.scatter(driver_stats['experience'], 
                     driver_stats['points'],
                     c=driver_stats['Cluster'],
                     cmap='viridis',
                     s=100)
plt.xlabel('Опыт (количество гонок)')
plt.ylabel('Средние очки за гонку')
plt.title('Кластеры гонщиков: Опыт vs Очки')
plt.colorbar(scatter, label='Кластер')
plt.show()

# Анализ кластеров
print("\nСредние характеристики кластеров:")
cluster_means = driver_stats.groupby('Cluster')[features].mean().round(2)
print(cluster_means)

print("\nКоличество гонщиков в каждом кластере:")
print(driver_stats['Cluster'].value_counts())

# Вывод примеров гонщиков из каждого кластера
print("\nПримеры гонщиков из каждого кластера:")
for cluster in range(n_clusters):
    print(f"\nКластер {cluster}:")
    cluster_drivers = driver_stats[driver_stats['Cluster'] == cluster]
    print(f"Количество гонщиков: {len(cluster_drivers)}")
    print("Топ-5 гонщиков по опыту:")
    top_drivers = cluster_drivers.nlargest(5, 'experience')[['full_name', 'experience', 'points']]
    print(top_drivers.to_string(index=False))