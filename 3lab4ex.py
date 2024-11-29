import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

'''
Задание 4. Удалите дубликаты из вашего набора данных. Обработайте возможные пропущенные значения.
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
    df_cleared = df.drop_duplicates()
    df_cleared = df_cleared.dropna()
    print(f"{title}")

    print("1" if df.shape == df_cleared.shape else "0")




