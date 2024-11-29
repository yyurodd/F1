'''
Лабораторная работа 5 - Анализ числовых переменных в данных Формулы-1

Этот скрипт выполняет анализ числовых (количественных) переменных
в данных Формулы-1, таких как позиция на старте, количество кругов,
очки и другие численные показатели.

Основные задачи:
1. Объединение данных из разных CSV файлов
2. Очистка и подготовка числовых данных
3. Обработка нулевых значений
4. Создание визуализаций распределения числовых данных

Используемые наборы данных:
- drivers.csv: информация о гонщиках
- results.csv: результаты гонок
- qualifying.csv: данные квалификаций
- races.csv: информация о гонках
- circuits.csv: данные о трассах

Анализируемые числовые переменные:
- grid: стартовая позиция
- laps: количество пройденных кругов
- points: набранные очки
- dob: год рождения гонщика (преобразован из полной даты)

Результат:
Создание визуализаций, показывающих распределение числовых
показателей в данных Формулы-1, включая гистограммы и графики плотности.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Отключение предупреждений для чистоты вывода
warnings.simplefilter('ignore')
sns.set()

'''
Задание 5. Произведите анализ распределений переменных. Интерпретируйте полученные результаты.
'''

# Загрузка всех необходимых CSV файлов
drivers = pd.read_csv("AI/F1/drivers.csv")
results = pd.read_csv("AI/F1/results.csv")
qualifying = pd.read_csv("AI/F1/qualifying.csv")
races = pd.read_csv("AI/F1/races.csv")
circuits = pd.read_csv("AI/F1/circuits.csv")

# Создание объединенного датафрейма и удаление ненужных столбцов
df = pd.merge(drivers,results, on='driverId')
df = pd.merge(df,races, on='raceId')
df = pd.merge(df,circuits, on='circuitId')
df.drop(['url_x','number_y','positionText','position','url_y','name_y','url'],axis=1,inplace=True)

# Предобработка данных:
# - Извлечение года из даты рождения
df['dob'] = df['dob'].str[:4]
# - Замена нулевых значений на None для корректного анализа
df['grid'] = df['grid'].replace(0, None)
df['laps'] = df['laps'].replace(0, None)
# - Удаление дубликатов по ID гонщика и году рождения
df = df.drop_duplicates(subset=['driverId','dob'], keep='first')
df['points'] = df['points'].replace(0, None)

# Настройка отображения всех столбцов в pandas
pd.set_option('display.max_columns', None)
print(df.head())

# Выборка необходимых числовых переменных
dfDigital = df[['dob','grid','positionOrder','points','laps','year']]
#dfDigital = dfDigital.drop_duplicates(subset=['code', 'forename', 'surname', 'nationality'], keep='first')
#pd.set_option('display.max_rows', None)
#print(dfDigital.head(100))

# Создание визуализаций распределения числовых данных
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
