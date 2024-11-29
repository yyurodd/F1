import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

'''
Задание 3. Загрузите набор данных для первичного просмотра. Выведите голову таблицы, чтобы убедиться что все отображается корректно.
 Выведите информацию о наборе (info). Выведите уникальные значения по каждому столбцу.
'''

drivers = pd.read_csv("drivers.csv")
results = pd.read_csv("results.csv")
qualifying = pd.read_csv("qualifying.csv")
races = pd.read_csv("races.csv")
circuits = pd.read_csv("circuits.csv")


dataset = [drivers, circuits, qualifying, races, results]
titles = ['Drivers', 'Circuits', 'Qualifying', 'Races', 'Results']


pd.set_option('display.max_columns', None)

for title, df in zip(titles, dataset):
    print(f"**{title}**")
    print(df.head(5))
    print("\n")

for title, df in zip(titles, dataset):
    print(f"**{title}**")
    print(df.info())
    print("\n")
