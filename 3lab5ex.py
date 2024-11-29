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

driver_perfomance = pd.merge(drivers,results, on='driverId')
driver_perfomance = pd.merge(driver_perfomance,races, on='raceId')
driver_perfomance = pd.merge(driver_perfomance,circuits, on='circuitId')
driver_perfomance.drop(['dob','url_x','constructorId','fastestLapSpeed','statusId','name_x','date','time_y','url_y','fp1_date','fp1_time','fp2_date','fp2_time','fp3_date','fp3_time','quali_date','quali_time','sprint_date','sprint_time','name_y','url'],axis=1,inplace=True)
pd.set_option('display.max_columns', None)
#print(driver_perfomance.head())

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

