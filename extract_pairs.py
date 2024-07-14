import os
import shutil
import json
from collections import defaultdict
from PIL import Image
import numpy as np

# Задайте пути к вашим данным и выходной папке
data_dir = ''
output_dir = './save_par'
# Создание выходных директорий
images_dir = os.path.join(output_dir, 'images')
meshes_dir = os.path.join(output_dir, 'meshes')
os.makedirs(images_dir, exist_ok=True)
os.makedirs(meshes_dir, exist_ok=True)

# Путь к файлу с аннотациями
annotations_file = os.path.join(data_dir, 'pix3d.json')

# Загрузка аннотаций
with open(annotations_file, 'r') as f:
    annotations = json.load(f)

# Группировка аннотаций по категориям
category_dict = defaultdict(list)
for item in annotations:
    # Проверка наличия маски в аннотации
    if 'mask' in item:
        category_dict[item['category']].append(item)

# Проверка наличия категорий
if len(category_dict) == 0:
    print("Нет категорий с изображениями и масками")
else:
    # Определение необходимого количества пар для каждой категории
    num_categories = len(category_dict)
    pairs_per_category = 100 // num_categories

    # Счетчик извлеченных пар
    count = 0

    # Функция для применения маски к изображению
    def apply_mask(image_path, mask_path):
        image = Image.open(image_path).convert('RGBA')
        mask = Image.open(mask_path).convert('L')
        
        # Преобразование маски в альфа-канал
        alpha = Image.fromarray(np.array(mask))
        image.putalpha(alpha)
        return image

    # Проход по категориям и извлечение пар
    for category, items in category_dict.items():
        pairs_extracted = 0
        for item in items:
            if pairs_extracted >= pairs_per_category:
                break

            img_path = os.path.join(data_dir, item['img'])
            mesh_path = os.path.join(data_dir, item['model'])
            mask_path = os.path.join(data_dir, item['mask'])

            # Проверяем, существуют ли файлы
            if os.path.exists(img_path) and os.path.exists(mesh_path) and os.path.exists(mask_path):
                # Применяем маску к изображению
                masked_image = apply_mask(img_path, mask_path)
                masked_image.save(os.path.join(images_dir, f'{category}_{pairs_extracted}.png'), 'PNG')
                # Копируем 3D модели в выходные директории
                shutil.copy(mesh_path, os.path.join(meshes_dir, f'{category}_{pairs_extracted}.obj'))
                pairs_extracted += 1
                count += 1

    print(f'Извлечено {count} пар изображений и 3D сеток с масками из разных категорий.')
