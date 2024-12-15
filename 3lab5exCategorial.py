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


drivers = pd.read_csv("AI/F1/drivers.csv")
results = pd.read_csv("AI/F1/results.csv")
qualifying = pd.read_csv("AI/F1/qualifying.csv")
races = pd.read_csv("AI/F1/races.csv")
circuits = pd.read_csv("AI/F1/circuits.csv")


df = pd.merge(drivers,results, on='driverId')
df = pd.merge(df,races, on='raceId')
df = pd.merge(df,circuits, on='circuitId')
df.drop(['url_x','number_y','positionText','position','url_y','name_y','url'],axis=1,inplace=True)

df['code'] = df['code'].replace(r'\N', None)


pd.set_option('display.max_columns', None)
#print(df.head())

# Выделение категориальных переменных для анализа
# Удаление дубликатов для получения уникальных комбинаций гонщиков
dfCategorial = df[['code','forename','surname','nationality','location','country']]
dfCategorial = dfCategorial.drop_duplicates(subset=['code', 'forename', 'surname', 'nationality'], keep='first')

dfCategorial.to_csv('AI/F1/dfCategorial.csv', index=False)


pd.set_option('display.max_rows', None)
#print(dfCategorial.head(1000))

# Создание визуализаций распределения категориальных данных
plt.figure(figsize=(15.3, 7.5))

for i, column in enumerate(dfCategorial.columns):
    # Берем максимум 100 значений для каждого графика, если доступно
    sample_data = dfCategorial[column]#.sample(n=min(100, len(dfCategorial[column])), random_state=1)

    plt.subplot(2, 3, i + 1)  # Создаем подграфик для каждого столбца
    sns.histplot(sample_data, bins=30, kde=True)
    plt.title(f'Распределение для {column}')

    plt.ylabel('Частота')
    plt.xticks(ticks=[])

plt.tight_layout()
plt.show()