'''
Лабораторная работа 8 - Анализ корреляций в данных Формулы-1

Этот скрипт выполняет корреляционный анализ данных Формулы-1,
фокусируясь на взаимосвязи между стартовой позицией и финишной
позицией на Гран-при Великобритании 2024 года.

Основные задачи:
1. Объединение данных из разных CSV файлов
2. Очистка и подготовка данных для корреляционного анализа
3. Построение матрицы корреляции
4. Визуализация корреляционных зависимостей

Используемые наборы данных:
- drivers.csv: информация о гонщиках
- results.csv: результаты гонок
- qualifying.csv: данные квалификаций
- races.csv: информация о гонках
- circuits.csv: данные о трассах

Анализируемые переменные:
- grid: стартовая позиция
- positionOrder: финишная позиция
- другие числовые показатели гонки

Результат:
Создание тепловой карты корреляций, показывающей степень
взаимосвязи между различными числовыми показателями гонки,
особенно между стартовой и финишной позициями.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Отключение предупреждений для чистоты вывода
warnings.simplefilter('ignore')
sns.set()

'''Задание 8. Постройте матрицу корреляции параметров датасета.'''

'''В данном задании сделана матрица корреляции места на стартовой решетке и места на финише на гран-при Великобритании 2024'''

# Загрузка всех необходимых CSV файлов
drivers = pd.read_csv("AI/F1/drivers.csv")
results = pd.read_csv("AI/F1/results.csv")
qualifying = pd.read_csv("AI/F1/qualifying.csv")
races = pd.read_csv("AI/F1/races.csv")
circuits = pd.read_csv("AI/F1/circuits.csv")

# Создание объединенного датафрейма и удаление ненужных столбцов
df = pd.merge(drivers,results, on='driverId')
df = pd.merge(df,races, on='raceId')
df = pd.merge(df,circuits, on='circuitId')
df.drop(['url_x','number_y','positionText','position','url_y','name_y','url'],axis=1,inplace=True)

# Предобработка данных:
# - Извлечение года из даты рождения
df['dob'] = df['dob'].str[:4]
# - Замена нулевых значений на None
df['grid'] = df['grid'].replace(0, None)
df['laps'] = df['laps'].replace(0, None)
# - Удаление дубликатов
df = df.drop_duplicates(subset=['driverId','dob'], keep='first')
df['points'] = df['points'].replace(0, None)

# Настройка отображения всех столбцов в pandas
pd.set_option('display.max_columns', None)

# Выборка необходимых столбцов для корреляционного анализа
dfCorr = df[['dob','grid','positionOrder','points']]

# Построение матрицы корреляции
dfCorr = dfCorr.corr()

# Визуализация корреляционных зависимостей
plt.figure(figsize=(9, 6))
sns.heatmap(dfCorr, annot=True, cmap='coolwarm', square=True)
plt.title('Матрица корреляции')
plt.show()
print(dfCorr)
