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
