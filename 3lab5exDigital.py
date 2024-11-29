import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.simplefilter('ignore')
sns.set()

'''
Задание 5. Произведите анализ распределений переменных. Интерпретируйте полученные результаты.
'''

drivers = pd.read_csv("drivers.csv")
results = pd.read_csv("results.csv")
qualifying = pd.read_csv("qualifying.csv")
races = pd.read_csv("races.csv")
circuits = pd.read_csv("circuits.csv")

df = pd.merge(drivers,results, on='driverId')
df = pd.merge(df,races, on='raceId')
df = pd.merge(df,circuits, on='circuitId')
df.drop(['url_x','number_y','positionText','position','url_y','name_y','url'],axis=1,inplace=True)
df['dob'] = df['dob'].str[:4]
df['grid'] = df['grid'].replace(0, None)
df['laps'] = df['laps'].replace(0, None)
df = df.drop_duplicates(subset=['driverId','dob'], keep='first')
df['points'] = df['points'].replace(0, None)
pd.set_option('display.max_columns', None)
print(df.head())

dfDigital = df[['dob','grid','positionOrder','points','laps','year']]
#dfDigital = dfDigital.drop_duplicates(subset=['code', 'forename', 'surname', 'nationality'], keep='first')
#pd.set_option('display.max_rows', None)
#print(dfDigital.head(100))

plt.figure(figsize=(15.3, 7.5))

for i, column in enumerate(dfDigital.columns):
    sample_data = dfDigital[column]#.sample(n=min(100, len(dfDigital[column])), random_state=1)
    plt.subplot(2, 3, i + 1)  # Создаем подграфик для каждого столбца
    sns.histplot(sample_data, bins=30, kde=True)  # Гистограмма с ядерной оценкой плотности
    plt.title(f'Распределение для {column}')
    plt.ylabel('Частота')
    plt.xticks(ticks=[])

plt.tight_layout()
plt.show()  # Показываем графики для числовых переменных
