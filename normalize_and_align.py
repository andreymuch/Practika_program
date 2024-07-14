import trimesh
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
import os

# Функция для центрирования модели
def center_model(mesh):
    centroid = mesh.centroid
    mesh.vertices -= centroid
    return mesh

# Функция для нормализации модели по размеру
def normalize_model(mesh):
    scale_factor = 1.0 / np.max(mesh.extents)
    mesh.apply_scale(scale_factor)
    return mesh

# Функция для предварительного выравнивания модели с использованием PCA
def align_by_pca(mesh):
    pca = PCA(n_components=3)
    pca.fit(mesh.vertices)
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = pca.components_.T
    mesh.apply_transform(rotation_matrix)
    return mesh

# Функция для ICP (Iterative Closest Point) для совмещения моделей
def icp(A, B, max_iterations=200, tolerance=1e-7):
    A_h = np.ones((A.shape[0], 4))
    B_h = np.ones((B.shape[0], 4))
    A_h[:, :-1] = A
    B_h[:, :-1] = B

    prev_error = 0
    for i in range(max_iterations):
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(B_h[:, :-1])
        distances, indices = nbrs.kneighbors(A_h[:, :-1])
        T = np.mean(B_h[indices.flatten(), :-1] - A_h[:, :-1], axis=0)
        A_h[:, :-1] += T

        rotation, _ = R.align_vectors(B_h[indices.flatten(), :-1], A_h[:, :-1])
        A_h[:, :-1] = rotation.apply(A_h[:, :-1])

        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    return A_h[:, :-1]

# Функция для подвыборки точек из меша
def sample_points(mesh, num_points):
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points

# Функция для корректировки поворота на 180 градусов
def correct_orientation(mesh, reference_mesh):
    num_points = min(len(mesh.vertices), len(reference_mesh.vertices), 1000)
    sample_mesh = sample_points(mesh, num_points)
    sample_reference = sample_points(reference_mesh, num_points)

    min_distance = np.inf
    best_rotation = None

    for angle_x in [0, 180]:
        for angle_y in [0, 180]:
            for angle_z in [0, 180]:
                rotation = R.from_euler('xyz', [angle_x, angle_y, angle_z], degrees=True)
                rotated_vertices = rotation.apply(sample_mesh)
                distance = np.mean(np.linalg.norm(rotated_vertices - sample_reference, axis=1))
                if distance < min_distance:
                    min_distance = distance
                    best_rotation = rotation

    if best_rotation is not None:
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = best_rotation.as_matrix()
        mesh.apply_transform(rotation_matrix)
    return mesh

# Обход всех моделей в директориях
dataset_dir = r"E:\Dokument\program\Pracktica\ProgramPraktika\save_par\meshes"

#neural_net_dir = r"E:\Dokument\program\Pracktica\ProgramPraktika\3d_models_from_neural_networks\not normalized\Triposr_output"
#neural_net_dir = r"E:\Dokument\program\Pracktica\ProgramPraktika\3d_models_from_neural_networks\not normalized\real3d_output"
neural_net_dir = r"E:\Dokument\program\Pracktica\ProgramPraktika\3d_models_from_neural_networks\not normalized\instantmesh_output_dir"

#output_dataset_dir = r'E:\Dokument\program\Pracktica\ProgramPraktika\3d_models_from_neural_networks\normalized\TripoSR_3d_model\dataset'
#output_dataset_dir = r'E:\Dokument\program\Pracktica\ProgramPraktika\3d_models_from_neural_networks\normalized\Real3D_3d_model\dataset'
output_dataset_dir = r'E:\Dokument\program\Pracktica\ProgramPraktika\3d_models_from_neural_networks\normalized\Instantmesh_3d_model\dataset'


#output_nn_dir = r'E:\Dokument\program\Pracktica\ProgramPraktika\3d_models_from_neural_networks\normalized\TripoSR_3d_model\TripoSr'
#output_nn_dir = r'E:\Dokument\program\Pracktica\ProgramPraktika\3d_models_from_neural_networks\normalized\Real3D_3d_model\real3d'
output_nn_dir = r'E:\Dokument\program\Pracktica\ProgramPraktika\3d_models_from_neural_networks\normalized\Instantmesh_3d_model\instantmesh'

def process_models(dataset_path, nn_path, output_dataset_path, output_nn_path):
    print(f"Processing models: {dataset_path} and {nn_path}")
    # Загрузка моделей
    model_dataset = trimesh.load(dataset_path)
    model_nn = trimesh.load(nn_path)

    # Проверка и извлечение мешей из сцен
    if isinstance(model_dataset, trimesh.Scene):
        model_dataset = model_dataset.dump(concatenate=True)
    if isinstance(model_nn, trimesh.Scene):
        model_nn = model_nn.dump(concatenate=True)

    # Центрирование, нормализация и выравнивание с использованием PCA
    model_dataset = center_model(model_dataset)
    model_dataset = normalize_model(model_dataset)
    model_dataset = align_by_pca(model_dataset)

    model_nn = center_model(model_nn)
    model_nn = normalize_model(model_nn)
    model_nn = align_by_pca(model_nn)

    # Применение ICP
    aligned_model_nn = icp(model_nn.vertices, model_dataset.vertices)

    # Обновление вершин модели
    model_nn.vertices = aligned_model_nn

    # Корректировка ориентации
    model_nn = correct_orientation(model_nn, model_dataset)

    # Сохранение результатов
    print(f"Saving dataset model to: {output_dataset_path}")
    model_dataset.export(output_dataset_path)
    print(f"Saving neural network model to: {output_nn_path}")
    model_nn.export(output_nn_path)

# Процессинг всех файлов
for file in os.listdir(dataset_dir):
    dataset_model_path = os.path.join(dataset_dir, file)
    name = file.split(".")[0]
#    nn_model_path = os.path.join(neural_net_dir, name + '.obj', r'0\mesh.obj')
#    nn_model_path = os.path.join(neural_net_dir, name, r'0\mesh.obj')
    nn_model_path = os.path.join(neural_net_dir, name + '.obj', r'instant-mesh-base\meshes',name + '.obj')
    print(dataset_model_path, nn_model_path)


    if os.path.exists(nn_model_path) and os.path.exists(dataset_model_path):
        output_dataset_path = os.path.join(output_dataset_dir, name + '_aligned.obj')
        output_nn_path = os.path.join(output_nn_dir, name + '_aligned.obj')

        process_models(dataset_model_path, nn_model_path, output_dataset_path, output_nn_path)
    else:
        if not os.path.exists(nn_model_path):
            print(f"NN model not found: {nn_model_path}")
        if not os.path.exists(dataset_model_path):
            print(f"Dataset model not found: {dataset_model_path}")
