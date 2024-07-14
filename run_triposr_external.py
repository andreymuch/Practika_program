import os
import sys
import subprocess

# Путь к папке с изображениями
images_dir = 'E:\Dokument\program\Pracktica\Dataset_Pix3D\save_par\images'
# Путь к выходной папке для результатов TripoSR
triposr_output_dir = 'E:\Dokument\program\Pracktica\ProgramPraktika\Triposr_output'
# Путь к репозиторию TripoSR
triposr_repo_dir = 'E:\Dokument\program\Pracktica\TripoSR'

# Создание выходной папки, если она не существует
os.makedirs(triposr_output_dir, exist_ok=True)

# Добавление пути к TripoSR в sys.path
sys.path.append(triposr_repo_dir)

# Проходим по всем изображениям в папке
for img in os.listdir(images_dir):
    input_path = os.path.join(images_dir, img)
    output_path = os.path.join(triposr_output_dir, os.path.splitext(img)[0] + '.obj')

    # Проверка и создание директории для текущего изображения
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    command = [
        'python',
        os.path.join(triposr_repo_dir, 'run.py'),
        input_path,
        '--output',
        output_path
    ]
    subprocess.run(command)
