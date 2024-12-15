import pandas as pd

# Пример данных
data = {
    'driverId': [1, 1, 1],
    'dob': [1985, 1985, 1985],
    'resultId': [1, 27, 57],
    'raceId': [18, 19, 20],
    'grid': [1, 9, 3],
    'positionOrder': [1, 5, 13],
    'points': [10.0, 4.0, 0.0],
    'laps': [58, 56, 56],
    'milliseconds': [5690616, 5525103, None],
    'rank': [2, 3, 19],
    'statusId': [1, 1, 11],
    'date': [2008, 2008, 2009],
    'constructorRef': ['mclaren', 'mclaren', 'mercedes'],
    'age': [23, 23, 48],
    'experience': [18.0, 19.0, 20.0],
    'constructor_score': [9.566087414428647, 9.566087414428647, 9.566087414428647]
}

df = pd.DataFrame(data)

# Агрегируем данные
df_aggregated = df.groupby('driverId').agg({
    'dob': 'first',  # Сохраняем уникальное значение
    'resultId': lambda x: ','.join(map(str, x.unique())),  # Объединяем значения в строку
    'raceId': lambda x: ','.join(map(str, x.unique())),
    'grid': 'mean',
    'positionOrder': 'mean',
    'points': 'sum',  # Суммируем очки
    'laps': 'sum',
    'milliseconds': lambda x: ','.join(map(str, x)),
    'rank': 'mean',
    'statusId': lambda x: ','.join(map(str, x)),
    'date': lambda x: ','.join(map(str, x.unique())),
    'constructorRef': lambda x: ','.join(map(str, x.unique())),
    'age': 'max',
    'experience': 'max',
    'constructor_score': 'mean'
}).reset_index()

print(df_aggregated)