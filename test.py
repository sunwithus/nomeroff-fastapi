from nomeroff_net import pipeline
from nomeroff_net.tools import unzip

import os
# Путь для скачивания моделей
os.environ['TORCH_HOME'] = r'D:\_ASR_Numbers\nomeroff-net\torch_models'
import torch

# Инициализируем пайплайн
number_plate_detection_and_reading = pipeline("number_plate_detection_and_reading",
                                              image_loader="opencv")

# Запускаем распознавание
results = number_plate_detection_and_reading(['data/examples/oneline_images/20190525.jpg'])

# Распаковываем ВСЕ результаты, которые возвращает текущая версия
unpacked = unzip(results)

print(f"Количество распакованных значений: {len(unpacked)}")

# В зависимости от версии, результаты могут быть в разных индексах
# Обычно тексты номеров находятся в последнем элементе
texts = unpacked[-1]  # Берем последний элемент - там обычно тексты
print("Распознанные номера:", texts)

# Для наглядности выведем всё, чтобы понять структуру
print("\nПолные результаты:")
for i, val in enumerate(unpacked):
    print(f"[{i}]: {type(val)} - {val if i < 2 else '...'}")