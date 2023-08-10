import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from google.colab import files
from io import BytesIO
from PIL import Image

# Инициализация модели VGG16
model = keras.applications.VGG16()

# Загрузка файла изображения с помощью Google Colab
uploaded = files.upload()

# Открывается и отображается загруженное изображение
img = Image.open(BytesIO(uploaded[list(uploaded.keys())[0]]))
plt.imshow(img)

# Преобразование изображения в массив
img = np.array(img)

# Предварительная обработка изображения для использования с VGG16
x = keras.applications.vgg16.preprocess_input(img)

# Вывод формы массива x (преобразованного изображения)
print(x.shape)

# Расширение массива x в измерении (оси) 0
x = np.expand_dims(x, axis=0)

# Предсказание модели VGG16
res = model.predict(x)

# Вывод индекса класса с наибольшей вероятностью
print(np.argmax(res))
