import os
import sys
import subprocess

# Путь к папке с изображениями
images_dir = 'E:\Dokument\program\Pracktica\Dataset_Pix3D\save_par\images'
# Путь к выходной папке для результатов InstantMesh
instantmesh_output_dir = 'E:\Dokument\program\Pracktica\ProgramPraktika\instantmesh_output_dir'
# Путь к репозиторию InstantMesh
instantmesh_repo_dir = 'E:\Dokument\program\Pracktica\InstantMesh'

# Создание выходной папки, если она не существует
os.makedirs(instantmesh_output_dir, exist_ok=True)

# Добавление пути к InstantMesh в sys.path
sys.path.append(instantmesh_repo_dir)

# Проходим по всем изображениям в папке
for img in os.listdir(images_dir):
    input_path = os.path.join(images_dir, img)
    output_path = os.path.join(instantmesh_output_dir, os.path.splitext(img)[0] + '.obj')
    command = [
        'python',
        os.path.join(instantmesh_repo_dir, 'run.py'),
        os.path.join(instantmesh_repo_dir, 'configs/instant-mesh-base.yaml'),
        input_path,
        '--output',
        output_path
    ]
    subprocess.run(command)
