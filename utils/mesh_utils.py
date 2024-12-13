import os
import torch
import numpy as np
import pymeshlab
import shutil
from pytorch3d.io import load_obj
import torch.nn.functional as F

# 将pcd点云保存为obj文件
def save_pointcloud_obj(points, save_path, normals=None):
    with open(save_path, 'w') as f:
        # 写入顶点
        for point in points:
            f.write(f'v {point[0]} {point[1]} {point[2]}\n')
        
        # 如果提供了法向量，则写入法向量
        if normals is not None:
            for normal in normals:
                f.write(f'vn {normal[0]} {normal[1]} {normal[2]}\n')
def normalize_vert(vertices, box_size=1.0, return_cs=False):
    if isinstance(vertices, np.ndarray):
        vmax, vmin = vertices.max(0), vertices.min(0)
        center = (vmax + vmin) * 0.5
        scale = box_size / np.max(vmax - vmin)
    else:  # torch.tensor
        vmax, vmin = vertices.max(0)[0], vertices.min(0)[0]
        center = (vmax + vmin) * 0.5
        scale = box_size / torch.max(vmax - vmin)
    if return_cs:
        return (vertices - center) * scale, center, scale
    # new_vertices = (vertices - center) * scale
    # print(f'center: {center}, {vmax}, {vmin}, {new_vertices.max(0)[0]}, {new_vertices.min(0)[0]}')
    return (vertices - center) * scale

def load_mesh_from_obj(obj_path, device='cpu', auto_uv=True):
    vertex, faces, aux = load_obj(obj_path)
    vertices = vertex.clone().detach().to(torch.float32).to(device)
    f = faces.verts_idx.clone().detach().to(torch.int32).to(device)
    vt = None
    ft = None
    if aux.verts_uvs is not None:
        vt = aux.verts_uvs.clone().detach().to(torch.float32).to(device)
        ft = faces.textures_idx.clone().detach().to(torch.int32).to(device)
    elif auto_uv:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(obj_path)
        ms.compute_texcoord_parametrization_triangle_trivial_per_wedge(textdim=10000)
        os.makedirs('temp', exist_ok=True)
        ms.save_current_mesh('temp/temp.obj')
        _, faces_new, aux_new = load_obj('temp/temp.obj')
        vt = aux_new.verts_uvs.clone().detach().to(torch.float32).to(device)
        ft = faces_new.textures_idx.clone().detach().to(torch.int32).to(device)
        shutil.rmtree('temp')
    return vertices, f, vt, ft

def to_py3d_mesh(vertices, faces, normals=None):
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer.mesh.textures import TexturesVertex
    mesh = Meshes(verts=[vertices], faces=[faces], textures=None)
    if normals is None:
        normals = mesh.verts_normals_packed()
    # set normals as vertext colors
    mesh.textures = TexturesVertex(verts_features=[normals / 2 + 0.5])
    return mesh


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)

def tensor2variable(tensor, device):
    # [1,23,3,3]
    return torch.tensor(tensor, device=device, requires_grad=True)