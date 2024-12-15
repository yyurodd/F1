import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

'''
Задание 3. Загрузите набор данных для первичного просмотра. Выведите голову таблицы, чтобы убедиться что все отображается корректно.
Выведите информацию о наборе (info). Выведите уникальные значения по каждому столбцу.
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

pd.set_option('display.max_columns', None)

# Первый цикл: Вывод первых 5 строк каждой таблицы
# zip() объединяет два списка (titles и dataset) в пары (название таблицы, датафрейм)
for title, df in zip(titles, dataset):
    print(f"**{title}**")  # Выводим название таблицы в формате markdown
    print(df.head(5))      # Выводим первые 5 строк датафрейма
    print("\n")            # Добавляем пустую строку для читаемости

# Второй цикл: Вывод информации о структуре каждой таблицы
# df.info() показывает:
# - Количество строк и столбцов
# - Названия всех столбцов
# - Тип данных каждого столбца
# - Количество непустых значений
# - Использование памяти
for title, df in zip(titles, dataset):
    print(f"**{title}**")  # Выводим название таблицы в формате markdown
    print(df.info())       # Выводим подробную информацию о структуре таблицы
    print("\n")            # Добавляем пустую строку для читаемости
