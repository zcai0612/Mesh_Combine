import torch
import open3d as o3d
import numpy as np
from pytorch3d.structures import Pointclouds
from pytorch3d.ops import estimate_pointcloud_normals
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes

def poisson_reconstruction(points, normals=None, depth=9):
    """
    点云泊松重建
    Args:
        points: numpy array of shape (N, 3)
        depth: 重建深度
        scale: 重建尺度
    Returns:
        vertices: numpy array of shape (V, 3)
        faces: numpy array of shape (F, 3)
    """
    # 转换为o3d点云格式
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 估计法向量
    if normals is None:
        pcd.estimate_normals()
    else:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    
    # 泊松重建
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, n_threads=-1)
    
    # 确保mesh是watertight
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    
    # 转换为numpy数组
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    
    return vertices, faces