'''
Лабораторная работа 4 - Задача регрессии на данных Формулы-1

Этот скрипт решает задачу регрессии для предсказания финишной позиции гонщика
на основе различных параметров гонки.

'''

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

df.drop(['driverRef','number_x','code','forename','surname','nationality','url','constructorId','number_y',
    'positionText','position','time','fastestLap','fastestLapTime','fastestLapSpeed','round',
    'circuitId','name','year','time_race','url_race','fp1_date','fp1_time','fp2_date','fp2_time','fp3_date','fp3_time','quali_date','quali_time','sprint_date','sprint_time','circuitRef','name_circuit','url_circuit'
    ,'location','country','lat','lng','alt','name_constructor','url_constructor','nationality_constructor'],axis=1,inplace=True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



# Заменим простое кодирование команд на их средний результат
constructor_performance = df.groupby('constructorRef')['positionOrder'].mean()
df['constructor_score'] = df['constructorRef'].map(constructor_performance)



def remove_anomalies(df):
    # 1. Создаем признак отклонения от стартовой позиции
    df['position_deviation'] = abs(df['grid'] - df['positionOrder'])
    
    # 2. Критерии аномалий
    anomaly_conditions = (
        # Резкое падение позиции (более чем на 10 позиций)
        (df['position_deviation'] > 10) & 
        # Исключаем последние позиции
        (df['positionOrder'] > 15)
    )
    
    # 3. Удаляем аномальные записи
    cleaned_df = df[~anomaly_conditions].copy()
    
    # 4. Дополнительные фильтры
    # Исключаем сходы с трассы
    cleaned_df = cleaned_df[
        (cleaned_df['statusId'] != 'Retired') & 
        (cleaned_df['statusId'] != 'Accident')
    ]
    
    
    
    return cleaned_df

# Применение функции очистки данных
df = remove_anomalies(df)


features = ['grid', 'laps', 'age', 'constructor_score']  # входные параметры
target = 'positionOrder'     # целевая метка

# Разделение на входные параметры (X) и целевую метку (y)
X = df[features]
y = df[target]


# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Создание и обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred = model.predict(X_test)
y_pred = np.round(y_pred).astype(int).clip(1)
# Ограничиваем предсказания в диапазоне от 1 до максимальной позиции
y_pred = np.clip(y_pred, 1, df['positionOrder'].max())

# Оценка качества модели
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Результаты оценки модели:")
print(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")
print(f"Коэффициент детерминации (R²): {r2:.2f}")

# Визуализация результатов
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Реальная позиция')
plt.ylabel('Предсказанная позиция')
plt.title('Сравнение реальных и предсказанных позиций')

plt.tight_layout()
plt.show()

# Визуализация важности признаков
importance = pd.DataFrame({
    'Признак': features,
    'Важность': abs(model.coef_)
})
plt.figure(figsize=(8, 4))
sns.barplot(data=importance, x='Важность', y='Признак')
plt.title('Важность признаков в модели')
plt.tight_layout()
plt.show()


# Вывод коэффициентов модели
print("\nКоэффициенты модели:")
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"Свободный член: {model.intercept_:.4f}")
