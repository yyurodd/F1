'''
Лабораторная работа 5 - Анализ распределений переменных в данных Формулы-1

Этот скрипт выполняет углубленный анализ данных гонок Формулы-1, 
фокусируясь на производительности гонщиков на конкретной трассе.

Основные задачи:
1. Объединение данных из разных CSV файлов для создания полной картины
2. Очистка и подготовка данных (удаление ненужных столбцов)
3. Анализ результатов гонок на трассе Silverstone в 2024 году
4. Визуализация распределения времени круга и других показателей

Используемые наборы данных:
- drivers.csv: информация о гонщиках
- results.csv: результаты гонок
- qualifying.csv: данные квалификаций
- races.csv: информация о гонках
- circuits.csv: данные о трассах

Результат:
Создание визуализаций, показывающих распределение различных 
показателей производительности гонщиков на трассе Silverstone.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Отключение предупреждений для чистоты вывода
warnings.simplefilter('ignore')
sns.set()

'''
Задание 5. Произведите анализ распределений переменных. Интерпретируйте полученные результаты.
'''

# Загрузка всех необходимых CSV файлов
drivers = pd.read_csv("AI/F1/drivers.csv")
results = pd.read_csv("AI/F1/results.csv")
qualifying = pd.read_csv("AI/F1/qualifying.csv")
races = pd.read_csv("AI/F1/races.csv")
circuits = pd.read_csv("AI/F1/circuits.csv")

# Создание комплексного датафрейма с информацией о производительности гонщиков
# Объединение таблиц по соответствующим ID
driver_perfomance = pd.merge(drivers,results, on='driverId')
driver_perfomance = pd.merge(driver_perfomance,races, on='raceId')
driver_perfomance = pd.merge(driver_perfomance,circuits, on='circuitId')

# Удаление ненужных столбцов для упрощения анализа
driver_perfomance.drop(['dob','url_x','constructorId','fastestLapSpeed','statusId','name_x','date','time_y','url_y','fp1_date','fp1_time','fp2_date','fp2_time','fp3_date','fp3_time','quali_date','quali_time','sprint_date','sprint_time','name_y','url'],axis=1,inplace=True)
#print(driver_perfomance.head())

# Настройка отображения всех столбцов в pandas
pd.set_option('display.max_columns', None)
#print(driver_perfomance.head())

# Фильтрация данных для анализа гонок на Silverstone в 2024 году
# Сортировка по времени круга для анализа производительности
silverstone_2024 = driver_perfomance[(driver_perfomance['year'] == 2024) & (driver_perfomance['circuitRef'] == 'silverstone')].copy()
#silverstone_2024['fastestLapTime'] = silverstone_2024['fastestLapTime'].replace('\\N', np.nan)
#silverstone_2024 = silverstone_2024.dropna(subset=['fastestLapTime'])
silverstone_2024 = silverstone_2024.sort_values(by='fastestLapTime')

# Группировка по driverId_y, получение минимального fastestLapTime и соответствующего кода
min_laps = silverstone_2024.groupby('driverId').agg({'fastestLapTime': 'min', 'code': 'first'}).reset_index()
min_laps = min_laps.drop_duplicates(subset=['driverId', 'fastestLapTime'])
#print(min_laps)

# Создание столбчатой диаграммы
plt.figure(figsize=(12, 6))
sns.barplot(data=min_laps.sort_values(by='fastestLapTime'), x='code', y='fastestLapTime', palette='viridis')

plt.xlabel("Код гонщика")
plt.ylabel("Лучшее время круга")
plt.title("Лучшее время круга на Гран-при Великобритании 2024")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
