#отключение предупреждений
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#подключаем все необходимые библиотеки
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense

#задаем значения двух списков, где значения заданы в разных мерах измерения,
#но эквивалентны друг другу
#данные списки нужны для обучения нейронной сети
c = np.array([-50, -20, -5, 0, 3, 6, 10, 15, 23, 28, 40, 41, 43, 89, 103, 104])
f = np.array([-58, -4, 23, 32, 37.4, 42.8, 50, 59, 73.4, 82.4, 104, 105.8, 109.4, 192.2, 217.4, 219.2])

#определяем последовательную модель нейронной сети
model = keras.Sequential()

#добавляем в эту модель слой нейронов, состоящий из одного выходного нейрона, 
#имеющий ровно один вход и линейную активационную функцию
model.add(Dense(units=1, input_shape=(1,), activation="linear"))

#критерий качества
model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(0.1))

#используем метод fit для обучения
history = model.fit(c, f, epochs=1000, verbose=0)
print("Обучение завершилось")

#тест обученной нейронной сети
print(model.predict([100]))
print(model.get_weights())