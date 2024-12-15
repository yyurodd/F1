import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.simplefilter('ignore')
sns.set()

# Загрузка данных
drivers = pd.read_csv("AI/F1/drivers.csv")
results = pd.read_csv("AI/F1/results.csv")
qualifying = pd.read_csv("AI/F1/qualifying.csv")
races = pd.read_csv("AI/F1/races.csv")
circuits = pd.read_csv("AI/F1/circuits.csv")
constructors = pd.read_csv("AI/F1/constructors.csv")


# Объединение данных
df = pd.merge(drivers, results, on='driverId')
df = pd.merge(df, races, on='raceId', suffixes=('', '_race'))
df = pd.merge(df, circuits, on='circuitId', suffixes=('', '_circuit'))
df = pd.merge(df, constructors, on='constructorId', suffixes=('', '_constructor'))

df['dob'] = pd.to_datetime(df['dob']).dt.year
df['date'] = pd.to_datetime(df['date']).dt.year
df['age'] = df['date'] - df['dob']

# Комбинированный показатель опыта
df['experience'] = df.groupby('driverId')['raceId'].transform('rank')
df.drop(['driverRef','number_x','code','forename','surname','nationality','url','constructorId','number_y',
    'positionText','position','time','fastestLap','fastestLapTime','fastestLapSpeed','round',
    'circuitId','name','year','time_race','url_race','fp1_date','fp1_time','fp2_date','fp2_time','fp3_date','fp3_time','quali_date','quali_time','sprint_date','sprint_time','circuitRef','name_circuit','url_circuit'
    ,'location','country','lat','lng','alt','name_constructor','url_constructor','nationality_constructor'],axis=1,inplace=True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



# Заменим простое кодирование команд на их средний результат
constructor_performance = df.groupby('constructorRef')['positionOrder'].mean()
df['constructor_score'] = df['constructorRef'].map(constructor_performance)

numeric_columns = ['grid', 'positionOrder', 'points', 'laps', 'milliseconds', 'rank', 'experience', 'constructor_score']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Группировка данных по driverId
df_unique = df.groupby('driverId').agg({
    'dob': 'first',
    'resultId': 'first',
    'raceId': 'first',
    'grid': 'mean',
    'positionOrder': 'mean',
    'points': 'sum',
    'laps': 'sum',
    'milliseconds': 'mean',
    'rank': 'mean',
    'statusId': 'first',
    'date': 'first',
    'constructorRef': 'first',
    'age': 'first',
    'experience': 'max',
    'constructor_score': 'mean'
}).reset_index()

# Запись в CSV файл
df_unique.to_csv('unique_drivers.csv', index=False)