'''
Лабораторная работа 4 - Обработка данных Формулы-1

Этот скрипт выполняет очистку и обработку данных из CSV файлов Формулы-1.
Основные задачи:
1. Удаление дубликатов из наборов данных
2. Обработка пропущенных значений
3. Сравнение размеров датафреймов до и после очистки

Используемые наборы данных:
- drivers.csv: информация о гонщиках
- results.csv: результаты гонок
- qualifying.csv: данные квалификаций
- races.csv: информация о гонках
- circuits.csv: данные о трассах

Результат:
Для каждого датафрейма выводится "1", если размер не изменился после очистки
(нет дубликатов и пропущенных значений), и "0", если были найдены и удалены
проблемные данные.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

'''
Задание 4. Удалите дубликаты из вашего набора данных. Обработайте возможные пропущенные значения.
'''

# Загрузка всех необходимых CSV файлов
drivers = pd.read_csv("AI/F1/drivers.csv")
results = pd.read_csv("AI/F1/results.csv")
qualifying = pd.read_csv("AI/F1/qualifying.csv")
races = pd.read_csv("AI/F1/races.csv")
circuits = pd.read_csv("AI/F1/circuits.csv")

# Создание списков для удобного перебора датафреймов
dataset = [drivers, circuits, qualifying, races, results]
titles = ['Drivers', 'Circuits', 'Qualifying', 'Races', 'Results']

# Настройка отображения всех столбцов в pandas
pd.set_option('display.max_columns', None)

# Обработка каждого датафрейма:
# 1. Удаление дубликатов с помощью drop_duplicates()
# 2. Удаление строк с пропущенными значениями с помощью dropna()
# 3. Сравнение размеров до и после очистки
for title, df in zip(titles, dataset):
    df_cleared = df.drop_duplicates()
    df_cleared = df_cleared.dropna()
    print(f"{title}")

    print("1" if df.shape == df_cleared.shape else "0")
