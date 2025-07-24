# Кейс 4: Оптимизация логистической сети для снижения углеродного следа
**Выполнили: Пудова Елизавета, Житникова Валентина, Липатова Галина, Борисова Анастасия**

**Описание ситуации:** Международная логистическая компания объявила о стратегии по снижению своего углеродного следа на 50% к 2030 году. Для достижения этой цели компании необходимо оптимизировать свою логистическую сеть, маршруты доставки и используемый транспорт. У компании есть большие массивы данных о перевозках, включая информацию о маршрутах, типах транспорта, весе грузов, расходе топлива и т.д.

**Контекст ситуации:** Логистическая отрасль является одним из значительных источников выбросов CO2. В условиях растущего внимания к экологическим проблемам и ужесточения регуляторных требований, компании вынуждены искать пути снижения своего воздействия на окружающую среду. При этом необходимо соблюдать баланс между экологичностью, стоимостью и скоростью доставки.

**Задания для работы с кейсом:**

Провести анализ текущей логистической сети компании и рассчитать ее углеродный след
Разработать модель для расчета выбросов CO2 для различных типов транспорта и маршрутов
Создать оптимизационную модель, минимизирующую углеродный след
Разработать интерактивную геопространственную визуализацию текущей и оптимизированной логистической сети
Предложить план поэтапного внедрения оптимизированной логистической стратегии

**Датасет: Global Supply Chain Dataset**

**Включает данные:** маршруты, тип транспорта, вес грузов, расход топлива и выбросы CO2.

**Импортируем библиотеки**
````
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
import matplotlib.pyplot as plt
````
**Загружаем файл с данными**
````
data = pd.read_csv('logistics_dataset.csv')
````
**Случайным образом делаем подбвыборку данных и выводим её**
````
df = data.sample(frac=0.035, random_state=42)
df
````
**Выводим информацию о датасете**
````
df.info()
````
````
Index: 411076 entries, 5528535 to 5824761
Data columns (total 18 columns):
 #   Column                       Non-Null Count   Dtype  
---  ------                       --------------   -----  
 0   origin                       411076 non-null  object 
 1   destination                  411076 non-null  object 
 2   distance_km                  411076 non-null  float64
 3   transport_type               411076 non-null  object 
 4   transport_type_code          411076 non-null  int64  
 5   cargo_weight_tons            411076 non-null  float64
 6   commodity                    411076 non-null  object 
 7   commodity_code               411076 non-null  int64  
 8   year                         411076 non-null  int64  
 9   ton_km                       411076 non-null  float64
 10  emission_factor_g_tkm        411076 non-null  int64  
 11  co2_emissions_kg             411076 non-null  float64
 12  co2_per_liter                411076 non-null  float64
 13  fuel_consumption_liters      411076 non-null  float64
 14  co2_per_km                   411076 non-null  float64
 15  co2_per_ton_km               411076 non-null  float64
 16  fuel_efficiency_l_per_100km  411076 non-null  float64
 17  environmental_class          411076 non-null  object 
dtypes: float64(9), int64(4), object(5)
````
**Проверяем данные на наличие дубликатов**
````
duplicates = df.duplicated().sum()
duplicates
````
Было найдено 10 дубликатов.

**Проверяем данные на наличие пропусков**
````
df.isnull().sum()
````
````
origin                         0
destination                    0
distance_km                    0
transport_type                 0
transport_type_code            0
cargo_weight_tons              0
commodity                      0
commodity_code                 0
year                           0
ton_km                         0
emission_factor_g_tkm          0
co2_emissions_kg               0
co2_per_liter                  0
fuel_consumption_liters        0
co2_per_km                     0
co2_per_ton_km                 0
fuel_efficiency_l_per_100km    0
environmental_class            0
dtype: int64
````
Пропуски отсутствуют.

**Удаляем дубликаты и выводим новый датасет**
````
df_clean = df.drop_duplicates()
df_clean
````
**Проверяем данные на наличие аномальных значений**
````
neg_distance_km = df_clean[df_clean['distance_km'] < 0]
print("Количество строк, где расстояние маршрута между пунктами отправления и назначения в километрах меньше 0: ", len(neg_distance_km))

neg_cargo_weight_tons = df_clean[df_clean['cargo_weight_tons'] < 0]
print("Количество строк, где масса перевезённого груза в тоннах меньше 0: ", len(neg_cargo_weight_tons))

neg_ton_km = df_clean[df_clean['ton_km'] < 0]
print("Количество строк, где грузооборот меньше 0: ", len(neg_ton_km))

neg_emission_factor_g_tkm = df_clean[df_clean['emission_factor_g_tkm'] < 0]
print("Количество строк, где удельные выбросы CO₂ по виду транспорта меньше 0: ", len(neg_emission_factor_g_tkm))

neg_co2_emissions_kg = df_clean[df_clean['co2_emissions_kg'] < 0]
print("Количество строк, где общие выбросы углекислого газа по маршруту меньше 0: ", len(neg_co2_emissions_kg))

neg_co2_per_liter = df_clean[df_clean['co2_per_liter'] < 0]
print("Количество строк, где масса CO₂, выделяемая при сгорании одного литра топлива меньше 0: ", len(neg_co2_per_liter))

neg_fuel_consumption_liters = df_clean[df_clean['fuel_consumption_liters'] < 0]
print("Количество строк, где оценочный объём потреблённого топлива на всём маршруте меньше 0: ", len(neg_fuel_consumption_liters))

neg_co2_per_km = df_clean[df_clean['co2_per_km'] < 0]
print("Количество строк, где интенсивность выбросов CO₂ на каждый километр маршрута меньше 0: ", len(neg_co2_per_km))

neg_co2_per_ton_km = df_clean[df_clean['co2_per_ton_km'] < 0]
print("Количество строк, где интенсивность выбросов CO₂ на одну тонну на километр меньше 0: ", len(neg_co2_per_ton_km))

neg_fuel_efficiency_l_per_100km = df_clean[df_clean['fuel_efficiency_l_per_100km'] < 0]
print("Количество строк, где средний расход топлива на 100 км маршрута  меньше 0: ", len(neg_fuel_efficiency_l_per_100km))
````
````
Количество строк, где расстояние маршрута между пунктами отправления и назначения в километрах меньше 0:  0
Количество строк, где масса перевезённого груза в тоннах меньше 0:  0
Количество строк, где грузооборот меньше 0:  0
Количество строк, где удельные выбросы CO₂ по виду транспорта меньше 0:  0
Количество строк, где общие выбросы углекислого газа по маршруту меньше 0:  0
Количество строк, где масса CO₂, выделяемая при сгорании одного литра топлива меньше 0:  0
Количество строк, где оценочный объём потреблённого топлива на всём маршруте меньше 0:  0
Количество строк, где интенсивность выбросов CO₂ на каждый километр маршрута меньше 0:  0
Количество строк, где интенсивность выбросов CO₂ на одну тонну на километр меньше 0:  0
Количество строк, где Средний расход топлива на 100 км маршрута  меньше 0:  0
````
Аномальных значений не найдено.

**Выведем информацию о количестве маршрутов и периоде данных, типах транспорта и их распределении**
````
print(f"Всего маршрутов: {len(df_clean)}")
print(f"Период данных: {df_clean['year'].min()}-{df_clean['year'].max()}")
print("\nТипы транспорта и их распределение:")
print(df_clean['transport_type'].value_counts())
````
````
Всего маршрутов: 411066
Период данных: 2018-2024

Типы транспорта и их распределение:
transport_type
Truck                      164016
Air (include truck-air)     96955
Multiple modes & mail       92543
Rail                        48104
Water                        6971
Other and unknown            1525
Pipeline                      952
````
**Проанализируем текущие выбросы CO2**
````
current_co2 = df_clean['co2_emissions_kg'].sum()
print(f"Текущий углеродный след: {current_co2:.2f} кг CO2 в год")
````
Текущий углеродный след: 83482.07 кг CO2 в год.

**Проанализируем типы транспорта**
````
co2_by_transport = df_clean.groupby('transport_type')['co2_emissions_kg'].sum().sort_values(ascending=False)
print("\nВыбросы CO2 по типам транспорта:")
print(co2_by_transport)
````
````
Выбросы CO2 по типам транспорта:
transport_type
Truck                      63931.798294
Rail                        7661.822742
Water                       5235.042736
Multiple modes & mail       2622.646072
Pipeline                    1611.843882
Air (include truck-air)     1580.749619
Other and unknown            838.162971
Name: co2_emissions_kg, dtype: float64
````
Таким образом больше всего выбросов CO2 идет от грузовиков, на втором месте - железнодорожный транспорт, на третьем - водный транспорт.

**Приведём статистику по основным показателям**
````
total_stats = {
    'total_co2_tonnes': df_clean['co2_emissions_kg'].sum() / 1000,
    'total_distance': df_clean['distance_km'].sum(),
    'total_cargo': df_clean['cargo_weight_tons'].sum(),
    'avg_emission_per_tonkm': df_clean['emission_factor_g_tkm'].mean()
}
total_stats
````
Совокупные выбросы CO2 по всем маршрутам в кг составили 83.48206631559368.

Общая протяженность всех маршрутов в км составила 743661.9267906458.

Суммарный тоннаж грузоперевозок составил 4779292.663571.

Суммарный удельный выброс CO2 составил 168.24417246865468.

**Приведём статистику по типам транспорта**
````
transport_stats = df_clean.groupby('transport_type').agg({
    'co2_emissions_kg': ['sum', 'mean'],
    'distance_km': 'mean',
    'cargo_weight_tons': 'mean',
    'emission_factor_g_tkm': 'mean',
    'fuel_efficiency_l_per_100km': 'mean'
}).sort_values(('co2_emissions_kg', 'sum'), ascending=False)

transport_stats['co2_share'] = transport_stats[('co2_emissions_kg', 'sum')] / df_clean['co2_emissions_kg'].sum() * 100

transport_stats
````
Для грузовиков суммарные выбросы CO2 составили 63931.798294 кг, а средние - 0.389790 кг, среднее расстояние - 1.379118 км, средняя масса перевезенного груза - 18.296844 т, средние удельные выбросы - 71.0 гр-ткм, средний расход топлива на 100 км - 48.472983 л, занимает 76,58% в суммарном объеме выбросов CO2.

Для железнодорожного транспорта суммарные выбросы CO2 составили 7661.822742 кг, а средние - 0.159276 кг, среднее расстояние - 2.097203 км, средняя масса перевезенного груза - 8.427178 т, средние удельные выбросы - 16.0 гр-ткм, средний расход топлива на 100 км - 5.031151 л, занимает 9.177807% в суммарном объеме выбросов CO2.

Для водного транспорта суммарные выбросы CO2 составили 5235.042736 кг, а средние - 0.750974 кг, среднее расстояние - 1.787906 км, средняя масса перевезенного груза - 24.717408 т, средние удельные выбросы - 35.0 гр-ткм, средний расход топлива на 100 км - 27.906751 л, занимает 6.270859% в суммарном объеме выбросов CO2.

Для multiple modes and mail суммарные выбросы CO2 составили 2622.646072 кг, а средние - 0.028340 кг, среднее расстояние - 2.355982 км, средняя масса перевезенного груза - 1.864972 т, средние удельные выбросы - 12.0 гр-ткм, средний расход топлива на 100 км - 0.721925 л, занимает 3.141568% в суммарном объеме выбросов CO2.

Для трубопроводного транспорта суммарные выбросы CO2 составили 1611.843882 кг, а средние - 1.693113 кг, среднее расстояние - 0.687790 км, средняя масса перевезенного груза - 999.904098 т, средние удельные выбросы - 4.0 гр-ткм, средний расход топлива на 100 км - 143.355426 л, занимает 1.930767% в суммарном объеме выбросов CO2.

Для воздушного транспорта суммарные выбросы CO2 составили 1580.749619 кг, а средние - 0.016304 кг, среднее расстояние - 1.890047 км, средняя масса перевезенного груза - 0.016063 т, средние удельные выбросы - 570.0 гр-ткм, средний расход топлива на 100 км - 0.290656 л, занимает 1.893520% в суммарном объеме выбросов CO2.

Для других и неизвестных типов транспорта суммарные выбросы CO2 составили 838.162971 кг, а средние - 0.549615 кг, среднее расстояние - 1.431666 км, средняя масса перевезенного груза - 48.901836 т, средние удельные выбросы - 80.0 гр-ткм, средний расход топлива на 100 км - 145.975631 л, занимает 1.004004% в суммарном объеме выбросов CO2.

**Сделаем визуализацию текущих выбросов CO2 по типам транспорта**
````
fig = px.bar(co2_by_transport.reset_index(), 
             x='transport_type', y='co2_emissions_kg',
             title='Текущие выбросы CO2 по типам транспорта')
fig.show()
````
<img width="1288" height="386" alt="текущие выбросы" src="https://github.com/user-attachments/assets/371cd1bf-d6e5-4ee9-acf7-25b16484a341" />

**Построим график распределения выбросов по маршрутам**
<img width="1131" height="773" alt="распределение выбросов" src="https://github.com/user-attachments/assets/bbd66f91-77a7-4d0e-b053-93eb6c2dc05b" />

**Построим график зависимости выбросов от расстояния**
<img width="1143" height="774" alt="зависимость выбросов" src="https://github.com/user-attachments/assets/5d9cd266-e38c-4354-af95-53f2a6ff1df9" />

**Создадим новые признаки**
````
df_clean['load_factor'] = df_clean['cargo_weight_tons'] / df_clean.groupby('transport_type')['cargo_weight_tons'].transform('max')
df_clean['distance_category'] = pd.cut(df_clean['distance_km'], 
                                 bins=[0, 200, 500, 1000, np.inf],
                                 labels=['short', 'medium', 'long', 'extra_long'])
````
**Разделим данные на тестовую и обучающую выборки**
````
X = df_clean[['transport_type', 'distance_km', 'cargo_weight_tons', 'load_factor', 'distance_category']]
y = df_clean['co2_emissions_kg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
````
**Создадим пайплайн**
````
numeric_features = ['distance_km', 'cargo_weight_tons', 'load_factor']
categorical_features = ['transport_type', 'distance_category']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])
````
**Проведём подбор гиперпараметров**
````
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
````
**Выведем результаты**
````
GridSearchCV(cv=5,
             estimator=Pipeline(steps=[('preprocessor',
                                        ColumnTransformer(transformers=[('num',
                                                                         StandardScaler(),
                                                                         ['distance_km',
                                                                          'cargo_weight_tons',
                                                                          'load_factor']),
                                                                        ('cat',
                                                                         OneHotEncoder(handle_unknown='ignore'),
                                                                         ['transport_type',
                                                                          'distance_category'])])),
                                       ('regressor',
                                        RandomForestRegressor(random_state=42))]),
             n_jobs=-1,
             param_grid={'regressor__max_depth': [None, 10, 20],
                         'regressor__min_samples_split': [2, 5],
                         'regressor__n_estimators': [100, 200]},
             scoring='neg_mean_absolute_error')
````
````
Pipeline(steps=[('preprocessor',
                 ColumnTransformer(transformers=[('num', StandardScaler(),
                                                  ['distance_km',
                                                   'cargo_weight_tons',
                                                   'load_factor']),
                                                 ('cat',
                                                  OneHotEncoder(handle_unknown='ignore'),
                                                  ['transport_type',
                                                   'distance_category'])])),
                ('regressor', RandomForestRegressor(random_state=42))])
````
````
ColumnTransformer(transformers=[('num', StandardScaler(),
                                 ['distance_km', 'cargo_weight_tons',
                                  'load_factor']),
                                ('cat', OneHotEncoder(handle_unknown='ignore'),
                                 ['transport_type', 'distance_category'])])
````
````
num: ['distance_km', 'cargo_weight_tons', 'load_factor']
cat: ['transport_type', 'distance_category']
````
````
StandardScaler: StandardScaler()
````
````
OneHotEncoder: OneHotEncoder(handle_unknown='ignore')
````
````
RandomForestRegressor: RandomForestRegressor(random_state=42)
````
**Оценим и сохраним модель**
````
model = grid_search.best_estimator_
y_pred = model.predict(X_test)

print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f} кг")
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

joblib.dump(model, 'co2_emission_model.pkl')
````
````
MAE: 0.03 кг
R2 Score: 0.69
['co2_emission_model.pkl']
````
