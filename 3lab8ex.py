import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


warnings.simplefilter('ignore')
sns.set()

'''Задание 8. Постройте матрицу корреляции параметров датасета.'''



drivers = pd.read_csv("AI/F1/drivers.csv")
results = pd.read_csv("AI/F1/results.csv")
qualifying = pd.read_csv("AI/F1/qualifying.csv")
races = pd.read_csv("AI/F1/races.csv")
circuits = pd.read_csv("AI/F1/circuits.csv")


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



pd.set_option('display.max_columns', None)

# Выборка необходимых столбцов для корреляционного анализа
dfCorr = df[['dob','grid','positionOrder','laps']]


dfCorr = dfCorr.corr()


plt.figure(figsize=(9, 6))
sns.heatmap(dfCorr, annot=True, cmap='coolwarm', square=True)
plt.title('Матрица корреляции')
plt.show()
print(dfCorr)
