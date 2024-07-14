import os
import sys
import subprocess

# Путь к папке с изображениями
images_dir = 'E:\Dokument\program\Pracktica\Dataset_Pix3D\save_par\images'
# Путь к выходной папке для результатов Real3D
real3d_output_dir = 'E:\Dokument\program\Pracktica\ProgramPraktika\\real3d_output'
# Путь к репозиторию Real3D
real3d_repo_dir = 'E:\Dokument\program\Pracktica\Real3D'


# Создание выходной папки, если она не существует
os.makedirs(real3d_output_dir, exist_ok=True)

# Проходим по всем изображениям в папке
for img in os.listdir(images_dir):
    input_path = os.path.join(images_dir, img)
    output_subdir = os.path.join(real3d_output_dir, os.path.splitext(img)[0])
    
    # Проверка и создание директории для текущего изображения
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)
    
    command = [
        'python', 
        os.path.join(real3d_repo_dir, 'run.py'), 
        input_path,
        '--output-dir', 
        output_subdir,
        '--render',
    ]
    subprocess.run(command)