'''
Лабораторная работа - Применение нейронных сетей для прогнозирования позиции в гонке

Задача: Прогнозирование финишной позиции гонщика (задача регрессии)
Обоснование выбора:
1. У нас есть четкая целевая переменная (positionOrder в results)
2. Существует зависимость между множеством факторов и финишной позицией
3. Задача имеет практическую ценность для команд и анализа гонок
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from datetime import datetime

# Загрузка данных
results = pd.read_csv("AI/F1/results.csv")
races = pd.read_csv("AI/F1/races.csv")
drivers = pd.read_csv("AI/F1/drivers.csv")
constructors = pd.read_csv("AI/F1/constructors.csv")
circuits = pd.read_csv("AI/F1/circuits.csv")

# Подготовка данных
# Объединяем таблицы
race_data = pd.merge(results, races[['raceId', 'year', 'circuitId']], on='raceId')
race_data = pd.merge(race_data, drivers[['driverId', 'dob']], on='driverId')
race_data = pd.merge(race_data, constructors[['constructorId', 'constructorRef']], on='constructorId')
race_data = pd.merge(race_data, circuits[['circuitId', 'circuitRef']], on='circuitId')

# Рассчитываем возраст гонщика на момент гонки
race_data['dob'] = pd.to_datetime(race_data['dob'])
race_data['age'] = race_data.apply(lambda x: x['year'] - x['dob'].year, axis=1)

# Рассчитываем опыт (количество предыдущих гонок)
race_data['experience'] = race_data.groupby('driverId').cumcount()

# Кодируем категориальные признаки
le_constructor = LabelEncoder()
le_circuit = LabelEncoder()
race_data['constructor_encoded'] = le_constructor.fit_transform(race_data['constructorRef'])
race_data['circuit_encoded'] = le_circuit.fit_transform(race_data['circuitRef'])

# Улучшим расчет рейтинга команд
constructor_stats = race_data.groupby('constructorRef').agg({
    'positionOrder': ['mean', 'count'],
    'points': ['sum', 'mean']  # Добавим очки для лучшей оценки
}).reset_index()
constructor_stats.columns = ['constructorRef', 'avg_position', 'races_count', 'total_points', 'avg_points']

# Нормализуем рейтинг команд с учетом очков и позиций
constructor_stats['position_score'] = 1 - (constructor_stats['avg_position'] / constructor_stats['avg_position'].max())
constructor_stats['points_score'] = constructor_stats['avg_points'] / constructor_stats['avg_points'].max()
constructor_stats['team_rating'] = (constructor_stats['position_score'] + constructor_stats['points_score']) / 2

# Добавляем рейтинг команд к основным данным
race_data = pd.merge(race_data, constructor_stats[['constructorRef', 'team_rating']], on='constructorRef')

# Выбираем признаки
features = [
    'grid',                 # стартовая позиция
    'laps',                 # количество кругов
    'age',                  # возраст гонщика
    'experience',           # опыт (количество гонок)
    'team_rating',         # рейтинг команды
    'circuit_encoded'       # трасса
]
target = 'positionOrder'

# Удаляем строки с пропущенными значениями
race_data = race_data.dropna(subset=features + [target])

# Удаляем строки с экстремальными значениями
race_data = race_data[race_data['positionOrder'] <= 20]
race_data = race_data[race_data['grid'] <= 20]

# Создаем X и y
X = race_data[features]
y = race_data[target]

# Добавляем веса для признаков
feature_weights = {
    'grid': 2.0,           # Стартовая позиция - очень важный фактор
    'team_rating': 1.8,    # Рейтинг команды - второй по важности
    'experience': 1.2,     # Опыт имеет среднее влияние
    'circuit_encoded': 1.0, # Трасса влияет умеренно
    'age': 0.8,           # Возраст менее важен
    'laps': 0.5           # Количество кругов наименее важно
}

# Применяем веса к входным данным
X_weighted = X.copy()
for feature, weight in feature_weights.items():
    X_weighted[feature] = X_weighted[feature] * weight

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_weighted, y, test_size=0.2, random_state=42)

# Нормализация данных с учетом весов
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Создание модели
model = Sequential([
    Dense(128, activation='relu', input_shape=(len(features),)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

# Компиляция модели
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Обучение модели
history = model.fit(
    X_train_scaled, 
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)



# Визуализация результатов обучения
plt.figure(figsize=(15, 5))

# График ошибки
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='MSE (обучение)')
plt.plot(history.history['val_loss'], label='MSE (валидация)')
plt.title('График ошибки')
plt.xlabel('Эпоха')
plt.ylabel('MSE')
plt.legend()

# График MAE
plt.subplot(1, 3, 2)
plt.plot(history.history['mae'], label='MAE (обучение)')
plt.plot(history.history['val_mae'], label='MAE (валидация)')
plt.title('График MAE')
plt.xlabel('Эпоха')
plt.ylabel('MAE')
plt.legend()

# График предсказаний
plt.subplot(1, 3, 3)
predictions = model.predict(X_test_scaled)
predictions = np.clip(np.round(predictions), 1, 20)  # Ограничиваем предсказания от 1 до 20
plt.scatter(y_test, predictions, alpha=0.1, label='Предсказания')
plt.plot([0, 20], [0, 20], 'r--', label='Идеальные предсказания')
plt.xlabel('Реальная позиция')
plt.ylabel('Предсказанная позиция')
plt.title('Предсказания vs Реальность')
plt.legend()

plt.tight_layout()
plt.show()



# Анализ предсказаний
print("\nПримеры предсказаний:")
sample_size = 10
indices = np.random.randint(0, len(X_test), sample_size)

print("\nФормат: Старт -> Предсказание (Реально) | Трасса, Команда (рейтинг)")
for idx in indices:
    start_pos = X_test.iloc[idx]['grid']
    circuit = race_data['circuitRef'][X_test.index[idx]]
    constructor = race_data['constructorRef'][X_test.index[idx]]
    team_rating = race_data['team_rating'][X_test.index[idx]]
    pred_pos = predictions[idx][0]
    real_pos = y_test.iloc[idx]
    print(f"Старт: {start_pos:.0f} -> Предсказано: {pred_pos:.0f} (Реально: {real_pos:.0f}) | {circuit}, {constructor} ({team_rating:.2f})")

# Статистика ошибок
errors = np.abs(predictions.flatten() - y_test)
print("\nСтатистика ошибок:")
print(f"Средняя ошибка (MAE): {np.mean(errors):.2f} позиций")
print(f"Медианная ошибка: {np.median(errors):.2f} позиций")
print(f"90% предсказаний имеют ошибку меньше: {np.percentile(errors, 90):.2f} позиций")

# Заменим анализ важности признаков
print("\nВажность признаков (с учетом весов):")
for feature, weight in feature_weights.items():
    correlation = abs(np.corrcoef(X[feature], y)[0,1])
    weighted_importance = correlation * weight
    print(f"{feature}: {weighted_importance:.3f} (корреляция: {correlation:.3f}, вес: {weight})")

# Добавим анализ предсказаний для разных команд
print("\nАнализ по командам:")
team_analysis = []
for constructor in race_data['constructorRef'].unique():
    team_data = race_data[race_data['constructorRef'] == constructor]
    if len(team_data) > 10:  # Анализируем только команды с достаточным количеством гонок
        avg_real = team_data['positionOrder'].mean()
        team_rating = team_data['team_rating'].iloc[0]
        team_analysis.append({
            'team': constructor,
            'avg_position': avg_real,
            'rating': team_rating,
            'races': len(team_data)
        })

team_analysis = pd.DataFrame(team_analysis)
team_analysis = team_analysis.sort_values('rating', ascending=False)
print("\nТоп-5 команд по рейтингу:")
print(team_analysis.head().to_string())