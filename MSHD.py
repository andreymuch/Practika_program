import trimesh
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def load_mesh(file_path):
    return trimesh.load(file_path)

def sample_points_on_mesh(mesh, num_samples):
    points, _ = trimesh.sample.sample_surface(mesh, num_samples)
    return points

def plot_mesh_with_points(mesh, points, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the mesh
    mesh_vertices = np.array(mesh.vertices)
    mesh_faces = np.array(mesh.faces)
    
    mesh_poly3d = Poly3DCollection(mesh_vertices[mesh_faces], alpha=0.5, facecolor='grey')
    ax.add_collection3d(mesh_poly3d)

    # Plot the points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')

    ax.set_title(title)
    plt.show()

def mean_symmetric_chamfer_distance(mesh1, mesh2, num_samples=10000):
    points1 = sample_points_on_mesh(mesh1, num_samples)
    points2 = sample_points_on_mesh(mesh2, num_samples)
    
    tree1 = KDTree(points1)
    tree2 = KDTree(points2)
    
    dists1, _ = tree1.query(points2)
    dists2, _ = tree2.query(points1)
    
    mean_dist1 = np.mean(dists1)
    mean_dist2 = np.mean(dists2)
    
    return mean_dist1, mean_dist2, points1, points2

# Загрузка моделей
mesh1 = load_mesh("E:\Dokument\program\Pracktica\ProgramPraktika\НЕудачная попытка\Pix3D_norma\Pix3D_normapath_to_save_normalized_pix3d_model.obj")
mesh2 = load_mesh(r"E:\Dokument\program\Pracktica\ProgramPraktika\НЕудачная попытка\Triposr_output_norma\normalized_model.obj")
# Вычисление метрик и получение точек
mean_dist1, mean_dist2, points1, points2 = mean_symmetric_chamfer_distance(mesh1, mesh2)

# Визуализация
plot_mesh_with_points(mesh1, points1, "Model 1 with Sampled Points")
plot_mesh_with_points(mesh2, points2, "Model 2 with Sampled Points")

print(f"Mean distance from model 1 to model 2: {mean_dist1}")
print(f"Mean distance from model 2 to model 1: {mean_dist2}")
