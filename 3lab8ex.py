import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.simplefilter('ignore')
sns.set()

'''Задание 8. Постройте матрицу корреляции параметров датасета.'''

'''В данном задании сделана матрица корреляции места на стартовой решетке и места на финише на гран-при Великобритании 2024'''

drivers = pd.read_csv("AI/F1/drivers.csv")
results = pd.read_csv("AI/F1/results.csv")
qualifying = pd.read_csv("AI/F1/qualifying.csv")
races = pd.read_csv("AI/F1/races.csv")
circuits = pd.read_csv("AI/F1/circuits.csv")


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

dfCorr = df[['dob','grid','positionOrder','points']]

dfCorr = dfCorr.corr()

plt.figure(figsize=(9, 6))
sns.heatmap(dfCorr, annot=True, cmap='coolwarm', square=True)
plt.title('Матрица корреляции')
plt.show()
print(dfCorr)
